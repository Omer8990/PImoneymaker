import os
import logging
from datetime import datetime
import asyncio
import signal
import ccxt
from telegram.ext import Application, CommandHandler
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
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
        self.grid_levels = 10
        self.grid_spread = 0.02  # 2% spread
        self.order_amount = 5
        self.min_profit_per_grid = 0.005
        self.active_orders = {}

        # Remove hardcoded price constraints and add dynamic ones
        self.price_buffer = 0.002  # 0.2% buffer for price movements
        self.last_price_update = None
        self.price_update_threshold = 0.01  # 1% price change triggers update

        # Risk management parameters
        self.max_position = 100
        self.min_usdt_reserve = 15
        self.min_order_size = {
            'PI': 1.0,
            'USDT': 5.0
        }

        # Performance tracking
        self.daily_stats = {
            'trades': 0,
            'profit': 0,
            'best_trade': 0,
            'worst_trade': 0
        }

        # Messaging templates
        self.grid_setup_message = (
            "🔲 Grid Trading Setup\n"
            "Grid Levels: {levels}\n"
            "Price Range: ${lower:.4f} - ${upper:.4f}\n"
            "Grid Spread: {spread}%\n"
            "Orders per Grid: {amount:.2f} PI\n"
            "Expected Profit per Grid: {profit}%"
        )

        self.trade_message = (
            "🔄 Grid Trade Executed\n"
            "Type: {type}\n"
            "Price: ${price:.4f}\n"
            "Amount: {amount:.2f} PI\n"
            "Grid Level: {level}/{total_levels}\n"
            "Profit: ${profit:.2f}"
        )

    async def calculate_grid_prices(self):
        """Calculate grid price levels with dynamic price range"""
        try:
            price_range = await self.get_current_price_range()
            if not price_range:
                return None

            lower_price = price_range['lower']
            upper_price = price_range['upper']

            # Calculate price levels
            grid_prices = []
            price_step = (upper_price - lower_price) / self.grid_levels

            for i in range(self.grid_levels + 1):
                price = lower_price + (price_step * i)
                grid_prices.append(price)

            return grid_prices
        except Exception as e:
            logger.error(f"Error calculating grid prices: {str(e)}")
            return None

    async def get_current_price_range(self):
        """Get current price range based on market conditions"""
        try:
            ticker = self.exchange.fetch_ticker(self.symbol)
            current_price = ticker['last']

            # Calculate dynamic price range
            lower_price = current_price * (1 - self.grid_spread / 2)
            upper_price = current_price * (1 + self.grid_spread / 2)

            # Add small buffer to prevent immediate invalidation
            lower_price = lower_price * (1 - self.price_buffer)
            upper_price = upper_price * (1 + self.price_buffer)

            return {
                'current': current_price,
                'lower': lower_price,
                'upper': upper_price
            }
        except Exception as e:
            logger.error(f"Error getting current price range: {str(e)}")
            return None

    async def calculate_optimal_order_sizes(self):
        """Calculate optimal order sizes based on available balances"""
        try:
            balances = await self.get_balances()

            # Available amounts after reserves
            usdt_available = balances['USDT'] - self.min_usdt_reserve
            pi_available = balances['PI']

            # Get current price
            ticker = self.exchange.fetch_ticker(self.symbol)
            current_price = ticker['last']

            # Calculate how many orders we can place on each side
            # For buy orders: USDT available / current price / number of grid levels
            max_buy_amount = (usdt_available / current_price) / (self.grid_levels / 2)

            # For sell orders: PI available / number of grid levels
            max_sell_amount = pi_available / (self.grid_levels / 2)

            # Take the minimum to ensure balanced grid
            optimal_order_size = min(
                max_buy_amount,
                max_sell_amount,
                self.max_position / self.grid_levels  # Respect max position size
            )

            # Ensure it's above minimum order size
            optimal_order_size = max(optimal_order_size, self.min_order_size['PI'])

            return optimal_order_size
        except Exception as e:
            logger.error(f"Error calculating optimal order sizes: {str(e)}")
            return self.order_amount  # fallback to default

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

    async def setup_grid(self):
        """Initialize the grid trading setup with optimal order sizes"""
        try:
            # Calculate optimal order size
            optimal_order_size = await self.calculate_optimal_order_sizes()

            # Get grid prices
            grid_prices = await self.calculate_grid_prices()
            if not grid_prices:
                return

            # Cancel existing orders
            await self.cancel_all_orders()

            # Track total committed amounts
            total_usdt_committed = 0
            total_pi_committed = 0

            # Place grid orders
            for i in range(len(grid_prices) - 1):
                buy_price = grid_prices[i]
                sell_price = grid_prices[i + 1]

                # Calculate USDT needed for this buy order
                usdt_needed = buy_price * optimal_order_size
                pi_needed = optimal_order_size

                balances = await self.get_balances()
                usdt_available = balances['USDT'] - self.min_usdt_reserve - total_usdt_committed
                pi_available = balances['PI'] - total_pi_committed

                # Place buy order if enough USDT
                if usdt_available >= usdt_needed:
                    buy_order = await self.place_order('buy', buy_price, optimal_order_size, i)  # Added level
                    if buy_order:
                        total_usdt_committed += usdt_needed

                # Place sell order if enough PI
                if pi_available >= pi_needed:
                    sell_order = await self.place_order('sell', sell_price, optimal_order_size, i)  # Added level
                    if sell_order:
                        total_pi_committed += pi_needed

            # Log utilization statistics
            logger.info(f"Grid setup complete. Using {total_usdt_committed:.2f} USDT and {total_pi_committed:.2f} PI")
            await self.send_telegram_message(
                f"📊 Grid Utilization:\n"
                f"USDT: {total_usdt_committed:.2f}\n"
                f"PI: {total_pi_committed:.2f}\n"
                f"Order Size: {optimal_order_size:.2f}"
            )

        except Exception as e:
            logger.error(f"Error setting up grid: {str(e)}")

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

                # Check for filled orders
                open_orders = self.exchange.fetch_open_orders(self.symbol)
                open_order_ids = set(order['id'] for order in open_orders)
                filled_orders = set(self.active_orders.keys()) - open_order_ids

                for order_id in filled_orders:
                    order_info = self.active_orders[order_id]
                    filled_order = self.exchange.fetch_order(order_id, self.symbol)

                    if not order_info.get('level'):
                        logger.warning(f"Order {order_id} missing level information, calculating from price")
                        # Calculate level based on price if missing
                        grid_prices = await self.calculate_grid_prices()
                        if grid_prices:
                            order_price = order_info['price']
                            # Find closest price level
                            level = min(range(len(grid_prices)),
                                        key=lambda i: abs(grid_prices[i] - order_price))
                            order_info['level'] = level

                    # Calculate profit
                    executed_price = filled_order['average'] or filled_order['price']
                    amount = filled_order['filled']
                    profit = amount * (executed_price - order_info['price'])

                    # Update stats
                    self.daily_stats['trades'] += 1
                    self.daily_stats['profit'] += profit
                    self.daily_stats['best_trade'] = max(self.daily_stats['best_trade'], profit)
                    self.daily_stats['worst_trade'] = min(self.daily_stats['worst_trade'], profit)

                    # Send trade notification
                    await self.send_telegram_message(
                        self.trade_message.format(
                            type=order_info['type'].upper(),
                            price=executed_price,
                            amount=amount,
                            level=order_info['level'],
                            total_levels=self.grid_levels,
                            profit=profit
                        )
                    )

                    # Place new order on opposite side
                    await self.place_counter_order(order_info)

                    # Remove filled order from tracking
                    del self.active_orders[order_id]

                # Sleep to prevent API rate limiting
                await asyncio.sleep(10)

        except Exception as e:
            logger.error(f"Error monitoring grid: {str(e)}")

    async def place_counter_order(self, filled_order_info):
        """Place a new order after one is filled"""
        try:
            # Recalculate grid prices to ensure we're using current market conditions
            grid_prices = await self.calculate_grid_prices()
            if not grid_prices:
                logger.error("Failed to calculate grid prices for counter order")
                return

            # Get the level from filled order
            level = filled_order_info.get('level')
            if level is None:
                logger.error("No level information in filled order")
                return

            # Place opposite order
            order_type = 'sell' if filled_order_info['type'] == 'buy' else 'buy'

            # For buy orders that were filled, place sell order at next level up
            # For sell orders that were filled, place buy order at next level down
            if order_type == 'sell':
                if level + 1 < len(grid_prices):
                    price = grid_prices[level + 1]
                else:
                    price = grid_prices[level]  # Use same level if at top of grid
            else:  # buy order
                if level - 1 >= 0:
                    price = grid_prices[level - 1]
                else:
                    price = grid_prices[level]  # Use same level if at bottom of grid

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

            # Place the new order
            new_order = None
            if order_type == 'buy':
                new_order = self.exchange.create_limit_buy_order(
                    self.symbol,
                    self.order_amount,
                    price
                )
            else:
                new_order = self.exchange.create_limit_sell_order(
                    self.symbol,
                    self.order_amount,
                    price
                )

            if new_order:
                self.active_orders[new_order['id']] = {
                    'price': price,
                    'type': order_type,
                    'level': level
                }

                # Log successful counter order placement
                logger.info(f"Placed counter {order_type} order at {price}")
                await self.send_telegram_message(
                    f"📎 Counter Order Placed\n"
                    f"Type: {order_type.upper()}\n"
                    f"Price: {price:.4f}\n"
                    f"Amount: {self.order_amount}\n"
                    f"Grid Level: {level}"
                )

        except Exception as e:
            logger.error(f"Error placing counter order: {str(e)}")

    async def place_order(self, order_type, price, amount, level=None):
        """Updated place_order without hardcoded price limits"""
        try:
            current_range = await self.get_current_price_range()
            if not current_range:
                return None

            # Validate price is within current market range
            if order_type == 'sell' and price < current_range['lower']:
                logger.warning(f"Skipping sell order: price {price} below current range")
                return None
            if order_type == 'buy' and price > current_range['upper']:
                logger.warning(f"Skipping buy order: price {price} above current range")
                return None

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

            return order
        except Exception as e:
            logger.error(f"Error placing {order_type} order at price {price}: {str(e)}")
            return None


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
        """Cancel all open orders manually since OKX doesn't support cancelAllOrders()"""
        try:
            open_orders = self.exchange.fetch_open_orders(self.symbol)
            for order in open_orders:
                try:
                    self.exchange.cancel_order(order['id'], self.symbol)
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

    async def run(self):
        """Main run loop"""
        try:
            # Initial grid setup
            await self.setup_grid()

            # Start grid monitoring
            await self.monitor_grid()

        except Exception as e:
            logger.error(f"Error in main run loop: {str(e)}")

    async def stop(self):
        """Stop the trading bot gracefully"""
        logger.info("Stopping grid trading bot...")
        self.running = False
        await self.cancel_all_orders()
        if self.app:
            await self.send_telegram_message("🔴 Grid trading bot is shutting down...")


# Main function and signal handlers remain the same as original bot
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
    # Use the new asyncio API to avoid deprecation warning
    async def run_bot():
        try:
            await main()
        except Exception as e:
            logger.error(f"Fatal error: {str(e)}")

    asyncio.run(run_bot())

