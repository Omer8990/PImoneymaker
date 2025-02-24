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
        self.grid_levels = 8
        self.grid_spread = 0.12  # 5% spread
        self.order_amount = 5
        self.min_profit_per_grid = 0.005
        self.active_orders = {}

        # Remove hardcoded price constraints and add dynamic ones
        self.price_buffer = 0.0005  # Can reduce to 0.0005 for more frequent trades
        self.price_update_threshold = 0.01  # Can reduce to 0.01 for more frequent grid updates
        self.last_price_update = None

        # Risk management parameters
        self.max_position = 300
        self.min_usdt_reserve = 5
        self.min_order_size = {
            'PI': 3.0,
            'USDT': 10.0
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
        """Calculate grid price levels with improved price limit handling"""
        try:
            ticker = self.exchange.fetch_ticker(self.symbol)
            if not ticker or 'last' not in ticker:
                logger.error("Failed to get current price from ticker")
                return None

            current_price = ticker['last']

            # Get exchange info to determine exact price limits
            exchange_info = self.exchange.load_markets()
            symbol_info = exchange_info.get(self.symbol, {})

            # Set strict price limits based on current price
            # Assuming 5% maximum deviation for safety
            price_limit_range = 0.05
            min_price = current_price * (1 - price_limit_range)
            max_price = current_price * (1 + price_limit_range)

            # Calculate grid points around current price
            grid_prices = []

            # Calculate step size based on grid levels
            total_range = max_price - min_price
            step_size = total_range / (self.grid_levels - 1)

            # Generate base grid prices
            for i in range(self.grid_levels):
                price = min_price + (step_size * i)
                # Round to 4 decimal places to match exchange precision
                price = round(price, 4)
                grid_prices.append(price)

            # Adjust prices near boundaries
            validated_prices = []
            for price in grid_prices:
                # Add additional validation for sell orders
                if price < current_price:  # Buy orders
                    validated_prices.append(price)
                else:  # Sell orders
                    # Ensure sell orders are above minimum sell price
                    if price >= 1.615:  # Based on observed limits
                        validated_prices.append(price)

            # Remove duplicates and sort
            validated_prices = sorted(list(set(validated_prices)))

            # Ensure we have enough valid price levels
            if len(validated_prices) < 3:
                logger.warning("Not enough valid price levels generated")
                return None

            return validated_prices

        except Exception as e:
            logger.error(f"Error calculating grid prices: {str(e)}")
            return None

    async def get_current_price_range(self):
        """Get current price range based on market conditions with safe null handling"""
        try:
            # Get current price
            ticker = self.exchange.fetch_ticker(self.symbol)
            if not ticker or 'last' not in ticker:
                logger.error("Failed to get current price from ticker")
                return None

            current_price = ticker['last']
            if not current_price:
                logger.error("Current price is None")
                return None

            # Calculate initial price range
            lower_price = current_price * (1 - self.grid_spread / 2)
            upper_price = current_price * (1 + self.grid_spread / 2)

            try:
                # Safely get exchange limits
                exchange_info = self.exchange.load_markets()
                symbol_info = exchange_info.get(self.symbol, {})
                limits = symbol_info.get('limits', {})
                price_limits = limits.get('price', {})

                # Use safe default values if limits aren't available
                min_price = price_limits.get('min')
                max_price = price_limits.get('max')

                # Only apply limits if they exist
                if min_price is not None:
                    lower_price = max(lower_price, float(min_price))
                if max_price is not None:
                    upper_price = min(upper_price, float(max_price))
            except Exception as e:
                logger.warning(f"Could not fetch exchange limits, using calculated range: {str(e)}")
                # Continue with calculated range if exchange limits unavailable

            # Ensure we have valid prices after all calculations
            if lower_price <= 0 or upper_price <= 0 or lower_price >= upper_price:
                logger.error(f"Invalid price range calculated: lower={lower_price}, upper={upper_price}")
                return None

            # Add buffer proportional to the price range
            buffer_amount = (upper_price - lower_price) * self.price_buffer
            lower_price = lower_price + buffer_amount
            upper_price = upper_price - buffer_amount

            # Final validation
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
        """Enhanced grid setup with better error handling"""
        try:
            # Calculate optimal order size
            optimal_order_size = await self.calculate_optimal_order_sizes()

            # Get grid prices
            grid_prices = await self.calculate_grid_prices()
            if not grid_prices:
                logger.error("Failed to calculate grid prices")
                return

            # Cancel existing orders
            await self.cancel_all_orders()

            # Track committed amounts
            total_usdt_committed = 0
            total_pi_committed = 0

            # Get current price for reference
            ticker = self.exchange.fetch_ticker(self.symbol)
            current_price = ticker['last']

            # Place grid orders
            for i, price in enumerate(grid_prices):
                # Determine order type based on price relative to current price
                order_type = 'buy' if price < current_price else 'sell'

                # Calculate required amounts
                usdt_needed = price * optimal_order_size
                pi_needed = optimal_order_size

                # Check balances
                balances = await self.get_balances()
                usdt_available = balances['USDT'] - self.min_usdt_reserve - total_usdt_committed
                pi_available = balances['PI'] - total_pi_committed

                # Place order if enough balance
                if order_type == 'buy' and usdt_available >= usdt_needed:
                    order = await self.place_order('buy', price, optimal_order_size, i)
                    if order:
                        total_usdt_committed += usdt_needed
                elif order_type == 'sell' and pi_available >= pi_needed:
                    order = await self.place_order('sell', price, optimal_order_size, i)
                    if order:
                        total_pi_committed += pi_needed

            # Log setup completion
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
        """Place order with improved price limit handling"""
        try:
            # Round price to 4 decimal places
            price = round(price, 4)

            # Get current market price
            ticker = self.exchange.fetch_ticker(self.symbol)
            current_price = ticker['last']

            # Add specific price validations based on order type
            if order_type == 'sell':
                min_sell_price = 1.615  # Based on observed limits
                if price < min_sell_price:
                    logger.warning(f"Skipping sell order: price {price} below minimum sell price {min_sell_price}")
                    return None

            elif order_type == 'buy':
                max_buy_price = current_price * 1.03  # Maximum 3% above current price
                if price > max_buy_price:
                    logger.warning(f"Skipping buy order: price {price} above maximum buy price {max_buy_price}")
                    return None

            # Add small delay between orders
            await asyncio.sleep(0.5)

            # Place the order
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
                logger.info(f"Successfully placed {order_type} order at price {price}")

            return order

        except Exception as e:
            if "price limit" in str(e).lower():
                logger.warning(f"Price limit reached for {order_type} order at price {price}")
            else:
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
