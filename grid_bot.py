import os
import logging
from datetime import datetime
import asyncio
import ccxt
from telegram.ext import Application
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging with more detailed format
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class GridTradingBot:
    def __init__(self):
        self.exchange = ccxt.okx({
            'apiKey': os.getenv('OKX_API_KEY'),
            'secret': os.getenv('OKX_SECRET'),
            'password': os.getenv('OKX_PASSWORD')
        })

        self.app = None
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.symbol = 'PI/USDT'
        self.running = True

        # Grid trading parameters
        self.grid_levels = 6
        self.grid_spread = 0.20  # 20% spread
        self.order_amount = 10
        self.min_profit_per_grid = 0.015  # 1.5% minimum profit
        self.active_orders = {}

        # Price movement parameters
        self.price_buffer = 0.0003
        self.price_update_threshold = 0.008
        self.dynamic_price_limit = 0.03
        self.last_price_update = None

        # Risk management parameters
        self.max_position = 500
        self.min_usdt_reserve = 3
        self.min_order_size = {
            'PI': 5.0,
            'USDT': 15.0
        }

        # Performance tracking
        self.daily_stats = {
            'trades': 0,
            'profit': 0,
            'best_trade': 0,
            'worst_trade': 0,
            'win_rate': 0,
            'average_profit_per_trade': 0,
            'total_volume': 0,
            'daily_roi': 0
        }

        # Volatility tracking
        self.price_history = []
        self.volatility_threshold = 0.02
        self.max_volatility_multiplier = 1.5

        # Message templates
        self.grid_setup_message = (
            "🔲 Grid Trading Setup\n"
            "Grid Levels: {levels}\n"
            "Price Range: ${lower:.4f} - ${upper:.4f}\n"
            "Current Price: ${current:.4f}\n"
            "Grid Spread: {spread}%\n"
            "Orders per Grid: {amount:.2f} PI\n"
            "Available PI: {pi_balance:.2f}\n"
            "Available USDT: {usdt_balance:.2f}\n"
            "Expected Profit per Grid: {profit}%"
        )

        self.trade_message = (
            "🔄 Grid Trade Executed\n"
            "Type: {type}\n"
            "Price: ${price:.4f}\n"
            "Amount: {amount:.2f} PI\n"
            "Grid Level: {level}/{total_levels}\n"
            "Current Market Price: ${market_price:.4f}\n"
            "Profit: ${profit:.2f}"
        )

        self.performance_message = (
            "📊 Performance Update\n"
            "Total Trades: {trades}\n"
            "Total Profit: ${profit:.2f}\n"
            "Win Rate: {win_rate:.1f}%\n"
            "Avg Profit/Trade: ${avg_profit:.2f}\n"
            "Best Trade: ${best_trade:.2f}\n"
            "Worst Trade: ${worst_trade:.2f}\n"
            "Daily ROI: {roi:.2f}%\n"
            "24h Volume: ${volume:.2f}"
        )

    async def calculate_volatility(self):
        """Calculate current market volatility"""
        if len(self.price_history) < 2:
            return 0

        returns = [
            (self.price_history[i] - self.price_history[i - 1]) / self.price_history[i - 1]
            for i in range(1, len(self.price_history))
        ]
        return abs(sum(returns) / len(returns))

    async def adjust_grid_parameters(self):
        """Dynamically adjust grid parameters based on market conditions"""
        try:
            volatility = await self.calculate_volatility()

            if volatility > self.volatility_threshold:
                adjusted_spread = self.grid_spread * self.max_volatility_multiplier
                logger.info(
                    f"High volatility detected ({volatility:.2%}). Increasing grid spread to {adjusted_spread:.2%}")
                return {
                    'grid_spread': adjusted_spread,
                    'order_amount': self.order_amount * 0.8
                }
            else:
                return {
                    'grid_spread': self.grid_spread,
                    'order_amount': self.order_amount
                }
        except Exception as e:
            logger.error(f"Error adjusting grid parameters: {str(e)}")
            return None

    async def calculate_grid_prices(self):
        """Calculate grid price levels with dynamic price limits"""
        try:
            ticker = self.exchange.fetch_ticker(self.symbol)
            if not ticker or 'last' not in ticker:
                logger.error("Failed to get current price from ticker")
                return None

            current_price = ticker['last']
            logger.info(f"Current market price: {current_price}")

            # Store price for volatility calculation
            self.price_history.append(current_price)
            if len(self.price_history) > 24:  # Keep last 24 prices
                self.price_history.pop(0)

            # Get adjusted parameters based on volatility
            adjusted_params = await self.adjust_grid_parameters()
            if not adjusted_params:
                adjusted_params = {
                    'grid_spread': self.grid_spread,
                    'order_amount': self.order_amount
                }

            # Calculate price range
            price_limit_range = adjusted_params['grid_spread'] / 2
            min_price = current_price * (1 - price_limit_range)
            max_price = current_price * (1 + price_limit_range)

            logger.info(f"Calculated price range: {min_price} - {max_price}")

            # Generate grid prices
            grid_prices = []
            step_size = (max_price - min_price) / (self.grid_levels - 1)

            for i in range(self.grid_levels):
                price = min_price + (step_size * i)
                price = round(price, 4)
                grid_prices.append(price)

            logger.info(f"Generated grid prices: {grid_prices}")
            return grid_prices

        except Exception as e:
            logger.error(f"Error calculating grid prices: {str(e)}")
            return None

    async def calculate_optimal_order_sizes(self):
        """Calculate optimal order sizes based on available balances"""
        try:
            balances = await self.get_balances()

            usdt_available = (balances['USDT'] - self.min_usdt_reserve) * 0.8
            pi_available = balances['PI'] * 0.8

            ticker = self.exchange.fetch_ticker(self.symbol)
            current_price = ticker['last']

            max_buy_amount = (usdt_available / current_price) / (self.grid_levels / 2)
            max_sell_amount = pi_available / (self.grid_levels / 2)

            optimal_order_size = max(
                min(max_buy_amount, self.max_position / self.grid_levels),
                min(max_sell_amount, self.max_position / self.grid_levels)
            )

            return max(optimal_order_size, self.min_order_size['PI'])
        except Exception as e:
            logger.error(f"Error calculating optimal order sizes: {str(e)}")
            return self.order_amount

    async def place_order(self, order_type, price, amount, level=None):
        """Place order with dynamic price validation"""
        try:
            price = round(price, 4)

            ticker = self.exchange.fetch_ticker(self.symbol)
            current_price = ticker['last']

            logger.info(f"Attempting to place {order_type} order at price {price} (Market: {current_price})")

            price_diff_percentage = abs(price - current_price) / current_price
            if price_diff_percentage > self.dynamic_price_limit:
                logger.warning(
                    f"Skipping {order_type} order: price {price} too far from market price {current_price} "
                    f"(diff: {price_diff_percentage:.2%})"
                )
                return None

            balances = await self.get_balances()
            if order_type == 'sell' and balances['PI'] < amount:
                logger.warning(f"Insufficient PI balance. Required: {amount}, Available: {balances['PI']}")
                return None
            elif order_type == 'buy' and balances['USDT'] < (price * amount):
                logger.warning(
                    f"Insufficient USDT balance. Required: {price * amount}, Available: {balances['USDT']}"
                )
                return None

            await asyncio.sleep(0.5)

            order = None
            if order_type == 'buy':
                order = self.exchange.create_limit_buy_order(self.symbol, amount, price)
            else:
                order = self.exchange.create_limit_sell_order(self.symbol, amount, price)

            if order:
                self.active_orders[order['id']] = {
                    'price': price,
                    'type': order_type,
                    'level': level
                }
                logger.info(
                    f"Successfully placed {order_type} order at price {price} "
                    f"(Market: {current_price}, Level: {level})"
                )

            return order

        except Exception as e:
            logger.error(f"Error placing {order_type} order at price {price}: {str(e)}")
            return None

    async def setup_grid(self):
        """Enhanced grid setup with better balance management"""
        try:
            balances = await self.get_balances()
            logger.info(f"Current balances - PI: {balances['PI']}, USDT: {balances['USDT']}")

            optimal_order_size = await self.calculate_optimal_order_sizes()
            logger.info(f"Calculated optimal order size: {optimal_order_size}")

            grid_prices = await self.calculate_grid_prices()
            if not grid_prices:
                logger.error("Failed to calculate grid prices")
                return

            await self.cancel_all_orders()

            ticker = self.exchange.fetch_ticker(self.symbol)
            current_price = ticker['last']

            total_usdt_committed = 0
            total_pi_committed = 0

            for i, price in enumerate(grid_prices):
                order_type = 'buy' if price < current_price else 'sell'

                usdt_needed = price * optimal_order_size
                pi_needed = optimal_order_size

                usdt_available = balances['USDT'] - self.min_usdt_reserve - total_usdt_committed
                pi_available = balances['PI'] - total_pi_committed

                logger.info(
                    f"Level {i}: {order_type} order at {price} "
                    f"(Available USDT: {usdt_available}, Available PI: {pi_available})"
                )

                if order_type == 'buy' and usdt_available >= usdt_needed:
                    order = await self.place_order('buy', price, optimal_order_size, i)
                    if order:
                        total_usdt_committed += usdt_needed
                elif order_type == 'sell' and pi_available >= pi_needed:
                    order = await self.place_order('sell', price, optimal_order_size, i)
                    if order:
                        total_pi_committed += pi_needed

            await self.send_telegram_message(
                self.grid_setup_message.format(
                    levels=self.grid_levels,
                    lower=grid_prices[0],
                    upper=grid_prices[-1],
                    current=current_price,
                    spread=self.grid_spread * 100,
                    amount=optimal_order_size,
                    pi_balance=balances['PI'],
                    usdt_balance=balances['USDT'],
                    profit=self.min_profit_per_grid * 100
                )
            )

        except Exception as e:
            logger.error(f"Error setting up grid: {str(e)}")

    async def update_performance_stats(self, profit, trade_amount):
        """Update and report performance statistics"""
        try:
            self.daily_stats['trades'] += 1
            self.daily_stats['profit'] += profit
            self.daily_stats['total_volume'] += trade_amount

            if profit > 0:
                self.daily_stats['win_rate'] = (
                        (self.daily_stats['win_rate'] * (self.daily_stats['trades'] - 1) + 100) /
                        self.daily_stats['trades']
                )

            self.daily_stats['average_profit_per_trade'] = (
                    self.daily_stats['profit'] / self.daily_stats['trades']
            )

            balances = await self.get_balances()
            ticker = self.exchange.fetch_ticker(self.symbol)
            total_equity = balances['USDT'] + (balances['PI'] * ticker['last'])
            self.daily_stats['daily_roi'] = (self.daily_stats['profit'] / total_equity) * 100

            if self.daily_stats['trades'] % 10 == 0:
                await self.send_telegram_message(
                    self.performance_message.format(
                        trades=self.daily_stats['trades'],
                        profit=self.daily_stats['profit'],
                        win_rate=self.daily_stats['win_rate'],
                        avg_profit=self.daily_stats['average_profit_per_trade'],
                        best_trade=self.daily_stats['best_trade'],
                        worst_trade=self.daily_stats['worst_trade'],
                        roi=self.daily_stats['daily_roi'],
                        volume=self.daily_stats['total_volume']
                    )
                )

        except Exception as e:
            logger.error(f"Error updating performance stats: {str(e)}")


    async def monitor_grid(self):
        """Monitor and maintain grid orders"""
        try:
            while self.running:
                if await self.should_update_grid():
                    logger.info("Price moved significantly, updating grid...")
                    await self.setup_grid()
                    current_range = await self.get_current_price_range()
                    if current_range:
                        self.last_price_update = current_range['current']

                open_orders = self.exchange.fetch_open_orders(self.symbol)
                open_order_ids = set(order['id'] for order in open_orders)
                filled_orders = set(self.active_orders.keys()) - open_order_ids

                for order_id in filled_orders:
                    order_info = self.active_orders[order_id]
                    filled_order = self.exchange.fetch_order(order_id, self.symbol)

                    if not order_info.get('level'):
                        logger.warning(f"Order {order_id} missing level information, calculating from price")
                        grid_prices = await self.calculate_grid_prices()
                        if grid_prices:
                            order_price = order_info['price']
                            level = min(range(len(grid_prices)),
                                        key=lambda i: abs(grid_prices[i] - order_price))
                            order_info['level'] = level

                    executed_price = filled_order['average'] or filled_order['price']
                    amount = filled_order['filled']
                    profit = amount * (executed_price - order_info['price'])

                    # Update best/worst trade stats
                    self.daily_stats['best_trade'] = max(self.daily_stats['best_trade'], profit)
                    self.daily_stats['worst_trade'] = min(self.daily_stats['worst_trade'], profit)

                    # Update performance stats
                    await self.update_performance_stats(profit, amount * executed_price)

                    # Send trade notification
                    await self.send_telegram_message(
                        self.trade_message.format(
                            type=order_info['type'].upper(),
                            price=executed_price,
                            amount=amount,
                            level=order_info['level'],
                            total_levels=self.grid_levels,
                            market_price=executed_price,
                            profit=profit
                        )
                    )

                    # Place new order on opposite side
                    await self.place_counter_order(order_info)

                    # Remove filled order from tracking
                    del self.active_orders[order_id]

                await asyncio.sleep(10)

        except Exception as e:
            logger.error(f"Error monitoring grid: {str(e)}")


    async def place_counter_order(self, filled_order_info):
        """Place a new order after one is filled"""
        try:
            grid_prices = await self.calculate_grid_prices()
            if not grid_prices:
                logger.error("Failed to calculate grid prices for counter order")
                return

            level = filled_order_info.get('level')
            if level is None:
                logger.error("No level information in filled order")
                return

            order_type = 'sell' if filled_order_info['type'] == 'buy' else 'buy'

            if order_type == 'sell':
                if level + 1 < len(grid_prices):
                    price = grid_prices[level + 1]
                else:
                    price = grid_prices[level]
            else:
                if level - 1 >= 0:
                    price = grid_prices[level - 1]
                else:
                    price = grid_prices[level]

            # Get current price range for validation
            current_range = await self.get_current_price_range()
            if not current_range:
                logger.error("Failed to get current price range for counter order")
                return

            # Validate price is within current market range
            if order_type == 'sell' and price < current_range['lower']:
                logger.warning(f"Skipping counter sell order: price {price} below current range")
                return
            if order_type == 'buy' and price > current_range['upper']:
                logger.warning(f"Skipping counter buy order: price {price} above current range")
                return

            # Calculate optimal order size
            optimal_order_size = await self.calculate_optimal_order_sizes()

            # Place the new order
            new_order = await self.place_order(order_type, price, optimal_order_size, level)

            if new_order:
                logger.info(f"Placed counter {order_type} order at {price}")
                await self.send_telegram_message(
                    f"📎 Counter Order Placed\n"
                    f"Type: {order_type.upper()}\n"
                    f"Price: {price:.4f}\n"
                    f"Amount: {optimal_order_size}\n"
                    f"Grid Level: {level}"
                )

        except Exception as e:
            logger.error(f"Error placing counter order: {str(e)}")


    async def get_current_price_range(self):
        """Get current price range based on market conditions"""
        try:
            ticker = self.exchange.fetch_ticker(self.symbol)
            if not ticker or 'last' not in ticker:
                logger.error("Failed to get current price from ticker")
                return None

            current_price = ticker['last']
            if not current_price:
                logger.error("Current price is None")
                return None

            # Get adjusted parameters based on volatility
            adjusted_params = await self.adjust_grid_parameters()
            if not adjusted_params:
                adjusted_params = {
                    'grid_spread': self.grid_spread,
                    'order_amount': self.order_amount
                }

            spread = adjusted_params['grid_spread']
            lower_price = current_price * (1 - spread / 2)
            upper_price = current_price * (1 + spread / 2)

            if lower_price <= 0 or upper_price <= 0 or lower_price >= upper_price:
                logger.error(f"Invalid price range calculated: lower={lower_price}, upper={upper_price}")
                return None

            buffer_amount = (upper_price - lower_price) * self.price_buffer
            lower_price = lower_price + buffer_amount
            upper_price = upper_price - buffer_amount

            if lower_price >= upper_price:
                logger.error("Price range invalid after adding buffer")
                return None

            return {
                'current': current_price,
                'lower': lower_price,
                'upper': upper_price
            }
        except Exception as e:
            logger.error(f"Error getting current price range: {str(e)}")
            return None


    async def should_update_grid(self):
        """Check if grid should be updated based on price movement"""
        try:
            current_range = await self.get_current_price_range()
            if not current_range or not self.last_price_update:
                return True

            price_change = abs(current_range['current'] - self.last_price_update) / self.last_price_update
            return price_change > self.price_update_threshold

        except Exception as e:
            logger.error(f"Error checking grid update: {str(e)}")
            return False


    async def get_balances(self):
        """Get both PI and USDT balances"""
        try:
            balance = self.exchange.fetch_balance()
            return {
                'PI': float(balance.get('PI', {}).get('free', 0)),
                'USDT': float(balance.get('USDT', {}).get('free', 0))
            }
        except Exception as e:
            logger.error(f"Error fetching balances: {str(e)}")
            return {'PI': 0, 'USDT': 0}


    async def cancel_all_orders(self):
        """Cancel all open orders"""
        try:
            open_orders = self.exchange.fetch_open_orders(self.symbol)
            for order in open_orders:
                try:
                    self.exchange.cancel_order(order['id'], self.symbol)
                    await asyncio.sleep(0.1)  # Small delay between cancellations
                except Exception as e:
                    logger.error(f"Error canceling order {order['id']}: {str(e)}")
            self.active_orders = {}
        except Exception as e:
            logger.error(f"Error fetching open orders: {str(e)}")


    async def send_telegram_message(self, message):
        """Send message via Telegram"""
        try:
            if self.app and self.chat_id:
                timestamp = datetime.now().strftime("%H:%M:%S")
                formatted_message = f"⏰ {timestamp}\n{message}"
                await self.app.bot.send_message(chat_id=self.chat_id, text=formatted_message)
        except Exception as e:
            logger.error(f"Error sending Telegram message: {str(e)}")


    async def stop(self):
        """Stop the trading bot gracefully"""
        logger.info("Stopping grid trading bot...")
        self.running = False
        await self.cancel_all_orders()
        if self.app:
            await self.send_telegram_message("🔴 Grid trading bot is shutting down...")


    async def run(self):
        """Main run loop"""
        try:
            await self.setup_grid()
            await self.monitor_grid()
        except Exception as e:
            logger.error(f"Error in main run loop: {str(e)}")


async def main():
    trading_bot = None
    try:
        trading_bot = GridTradingBot()

        app = Application.builder().token(os.getenv('TELEGRAM_TOKEN')).build()
        trading_bot.app = app
        app.bot_data['trading_bot'] = trading_bot

        await app.initialize()
        await app.start()
        await app.updater.start_polling()

        await trading_bot.run()

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        if trading_bot:
            await trading_bot.stop()
            if trading_bot.app:
                await trading_bot.app.shutdown()


if __name__ == "__main__":
    async def run_bot():
        try:
            await main()
        except Exception as e:
            logger.error(f"Fatal error: {str(e)}")


    asyncio.run(run_bot())