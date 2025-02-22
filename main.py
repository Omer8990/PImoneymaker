import os
import logging
from datetime import datetime, timedelta
import asyncio
import signal
import ccxt
from telegram.ext import Application, CommandHandler
import pandas as pd
import numpy as np
from dotenv import load_dotenv

import random

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class TradingBot:
    def __init__(self):
        self.exchange = ccxt.okx({
            'apiKey': os.getenv('OKX_API_KEY'),
            'secret': os.getenv('OKX_SECRET'),
            'password': os.getenv('OKX_PASSWORD')
        })

        self.app = None
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.symbol = 'PI/USDT'
        self.loss_threshold = 50
        self.profit_target = 100  # New: Profit target
        self.trading_allowed = True
        self.running = True

        # Enhanced technical parameters
        self.short_window = 7  # Shortened for quicker reactions
        self.long_window = 21  # Changed to fibonacci number
        self.rsi_period = 14  # Standard RSI period
        self.rsi_oversold = 35  # More aggressive oversold
        self.rsi_overbought = 65  # More aggressive overbought

        # Advanced trading parameters
        self.trailing_stop_pct = 0.02  # 2% trailing stop
        self.min_trade_size = 5
        self.max_position_size = None
        self.initial_balance = None
        self.current_loss = 0
        self.daily_stats = {
            'trades': 0,
            'profit': 0,
            'best_trade': 0,
            'worst_trade': 0
        }
        # Add USDT management parameters
        self.min_usdt_reserve = 10  # Keep minimum 10 USDT as reserve
        self.max_usdt_per_trade = 100  # Maximum USDT to use per trade
        self.usdt_position_pct = 0.2  # Use 20% of available USDT per trade
        self.MIN_PI_ORDER = 1.0  # Set to 1 PI minimum
        self.MIN_USDT_ORDER = 5.0  # Set to 5 USDT minimum

        # Also increase the minimum USDT reserve to ensure we have enough for fees
        self.min_usdt_reserve = 15  # Increased from 10

        # Adjust USDT position percentage to ensure larger orders
        self.usdt_position_pct = 0.5  # Increased from 0.2 to use 50% of available USDT

        # Personality messages
        self.profit_messages = [
            "🎯 Boss! We just scored a sweet profit of {profit:.2f} USDT! Your investment is working hard for you! 💪",
            "🚀 BOOM! Another win in the books! We're up {profit:.2f} USDT. Keep crushing it! 🏆",
            "💎 Your diamond hands paid off! Profit secured: {profit:.2f} USDT. This is the way! 🌟",
            "🎨 Painting the charts green! Just locked in {profit:.2f} USDT profit. You're a natural! 🎨"
        ]

        self.loss_messages = [
            "📉 Slight setback, boss. We're down {loss:.2f} USDT. But remember, every dip is a future comeback story! 💪",
            "🎭 Market's playing hard to get. Lost {loss:.2f} USDT, but we're adjusting our strategy. Stay cool! 😎",
            "🌊 Small wave against us, -{loss:.2f} USDT. But we're built for the ocean, not just the waves! 🏄‍♂️",
            "🎮 Level challenge: -{loss:.2f} USDT. But like any good game, we learn and come back stronger! 🎮"
        ]

    async def get_market_sentiment(self):
        """Analyze overall market sentiment"""
        try:
            # Fetch data from multiple timeframes
            timeframes = ['1m', '5m', '15m']
            sentiment_score = 0

            for tf in timeframes:
                ohlcv = self.exchange.fetch_ohlcv(self.symbol, tf, limit=30)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

                # Calculate trend strength
                df['trend'] = np.where(df['close'] > df['open'], 1, -1)
                df['volume_weight'] = df['volume'] / df['volume'].mean()
                sentiment_score += (df['trend'] * df['volume_weight']).mean()

            return sentiment_score / len(timeframes)
        except Exception as e:
            logger.error(f"Error in market sentiment analysis: {str(e)}")
            return 0

    async def send_telegram_message(self, message, is_important=False):
        """
        Send formatted message via Telegram

        Args:
            message (str): The message to send
            is_important (bool): Whether to mark the message as important with special formatting
        """
        try:
            if self.app and self.chat_id:
                timestamp = datetime.now().strftime("%H:%M:%S")

                if is_important:
                    formatted_message = f"🔔 IMPORTANT UPDATE 🔔\n⏰ {timestamp}\n{message}"
                else:
                    formatted_message = f"⏰ {timestamp}\n{message}"

                await self.app.bot.send_message(
                    chat_id=self.chat_id,
                    text=formatted_message
                )
        except Exception as e:
            logger.error(f"Error sending Telegram message: {str(e)}")

    async def send_daily_report(self):
        """Enhanced daily report with USDT balance"""
        try:
            balances = await self.get_balances()
            report = (
                f"📊 Daily Trading Report 📊\n\n"
                f"Trades Made: {self.daily_stats['trades']} 🎯\n"
                f"Total Profit/Loss: {self.daily_stats['profit']:.2f} USDT\n"
                f"Best Trade: {self.daily_stats['best_trade']:.2f} USDT 🏆\n"
                f"Worst Trade: {self.daily_stats['worst_trade']:.2f} USDT 📉\n\n"
                f"Current Balances:\n"
                f"PI: {balances['PI']:.2f}\n"
                f"USDT: ${balances['USDT']:.2f}\n"
                f"Initial PI Balance: {self.initial_balance:.2f}\n\n"
                f"{'🌟 Profitable Day! Keep it up! 🚀' if self.daily_stats['profit'] > 0 else '💪 Tomorrow is a new opportunity! 🌅'}"
            )
            await self.send_telegram_message(report, is_important=True)

            # Reset daily stats
            self.daily_stats = {
                'trades': 0,
                'profit': 0,
                'best_trade': 0,
                'worst_trade': 0
            }
        except Exception as e:
            logger.error(f"Error sending daily report: {str(e)}")

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

    async def calculate_buy_amount(self, current_price):
        """Calculate how much PI we can buy with available USDT"""
        try:
            balances = await self.get_balances()
            usdt_available = balances['USDT'] - self.min_usdt_reserve

            logger.info(f"Calculating buy - Available USDT: {usdt_available}, Current price: {current_price}")

            if usdt_available <= self.MIN_USDT_ORDER:
                logger.info(f"Insufficient USDT. Available: {usdt_available}, Minimum needed: {self.MIN_USDT_ORDER}")
                return 0

            # Always try to use at least MIN_USDT_ORDER worth of USDT
            usdt_to_use = max(
                usdt_available * self.usdt_position_pct,
                self.MIN_USDT_ORDER * 2  # Double the minimum to ensure we clear requirements
            )

            # Cap at max trade size
            usdt_to_use = min(usdt_to_use, self.max_usdt_per_trade)

            # Calculate PI amount
            pi_amount = usdt_to_use / current_price

            # Force minimum PI amount if we can afford it
            if pi_amount < self.MIN_PI_ORDER:
                if usdt_available >= (self.MIN_PI_ORDER * current_price * 1.1):  # Add 10% buffer for fees
                    pi_amount = self.MIN_PI_ORDER
                else:
                    logger.info(f"Cannot meet minimum PI order size of {self.MIN_PI_ORDER}")
                    return 0

            # Round to 1 decimal place to ensure clean amounts
            final_amount = round(pi_amount, 1)
            logger.info(f"Calculated buy amount: {final_amount} PI worth {final_amount * current_price} USDT")

            return final_amount

        except Exception as e:
            logger.error(f"Error calculating buy amount: {str(e)}")
            return 0

    async def send_telegram_message(self, message):
        """Send message via Telegram"""
        try:
            if self.app and self.chat_id:
                await self.app.bot.send_message(chat_id=self.chat_id, text=message)
        except Exception as e:
            logger.error(f"Error sending Telegram message: {str(e)}")

    async def initialize(self):
        """Initialize with both PI and USDT balances"""
        try:
            balances = await self.get_balances()

            # Make sure we get valid balances
            if balances['PI'] is not None:
                self.initial_balance = float(balances['PI'])
            else:
                self.initial_balance = 0.0  # Set a default if we can't get the balance

            logger.info(f"Initialized with balances: PI: {balances['PI']}, USDT: {balances['USDT']}")
            logger.info(f"Initial PI balance set to: {self.initial_balance}")

            await self.send_telegram_message(
                f"🏦 Initial Balances:\n"
                f"PI: {balances['PI']:.4f}\n"
                f"USDT: ${balances['USDT']:.4f}\n\n"
                f"Ready to trade! 🚀"
            )
        except Exception as e:
            logger.error(f"Error in initialization: {str(e)}")
            self.initial_balance = 0.0  # Set default on error

    async def stop(self):
        """Stop the trading bot gracefully"""
        logger.info("Stopping trading bot...")
        self.running = False
        if self.app:
            try:
                await self.send_telegram_message("🔴 Trading bot is shutting down...")
            except Exception as e:
                logger.error(f"Error sending shutdown message: {str(e)}")

    def get_market_data(self):
        """Fetch market data from exchange"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, '1m', limit=200)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            logger.error(f"Error fetching market data: {str(e)}")
            return None

    def calculate_signals(self, df):
        """Calculate technical indicators"""
        try:
            # Moving averages
            df['SMA7'] = df['close'].rolling(window=self.short_window).mean()
            df['SMA21'] = df['close'].rolling(window=self.long_window).mean()

            # RSI calculation
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))

            # Volume analysis
            df['Volume_MA'] = df['volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['volume'] / df['Volume_MA']

            # Make sure all values are numeric
            df = df.fillna(0)

            return df
        except Exception as e:
            logger.error(f"Error calculating signals: {str(e)}")
            return None

    def should_buy(self, df):
        """Determine if we should buy"""
        try:
            last_row = df.iloc[-1]

            # More lenient conditions
            conditions = [
                last_row['RSI'] < 45,  # Less strict RSI
                last_row['SMA7'] > last_row['SMA21'],
                last_row['Volume_Ratio'] > 1.1  # Less strict volume requirement
            ]

            # Only require 1 condition to be met
            return sum(conditions) >= 1
        except Exception as e:
            logger.error(f"Error in buy signal calculation: {str(e)}")
            return False

    def should_sell(self, df):
        """Determine if we should sell"""
        try:
            last_row = df.iloc[-1]
            conditions = [
                last_row['RSI'] > self.rsi_overbought,
                last_row['SMA7'] < last_row['SMA21'],  # Changed from SMA30
                last_row['Volume_Ratio'] < 0.8
            ]
            return sum(conditions) >= 2
        except Exception as e:
            logger.error(f"Error in sell signal calculation: {str(e)}")
            return False


    def calculate_position_size(self, signal_strength):
        """Calculate position size based on signal strength"""
        base_size = self.min_trade_size
        if self.max_position_size is None:
            return base_size
        return min(base_size * signal_strength, self.max_position_size)

    async def execute_trade(self, side, signal_strength):
        """Execute trade with enhanced USDT management"""
        try:
            sentiment = await self.get_market_sentiment()
            ticker = self.exchange.fetch_ticker(self.symbol)
            current_price = ticker['last']

            if side == 'buy':
                amount = await self.calculate_buy_amount(current_price)
                if amount < self.MIN_PI_ORDER:
                    logger.info(f"Buy amount {amount} is below minimum {self.MIN_PI_ORDER}, skipping trade")
                    await self.send_telegram_message(
                        f"ℹ️ Buy skipped - amount {amount} below minimum {self.MIN_PI_ORDER} PI"
                    )
                    return None

                # Check if we have enough USDT
                usdt_needed = amount * current_price
                if usdt_needed < self.MIN_USDT_ORDER:
                    logger.info(f"USDT value {usdt_needed} is below minimum {self.MIN_USDT_ORDER}, skipping trade")
                    await self.send_telegram_message(
                        f"ℹ️ Buy skipped - USDT value {usdt_needed:.2f} below minimum {self.MIN_USDT_ORDER} USDT"
                    )
                    return None
            else:  # sell
                balances = await self.get_balances()
                amount = balances['PI']
                if amount < self.MIN_PI_ORDER:
                    logger.info(f"Sell amount {amount} is below minimum {self.MIN_PI_ORDER}, skipping trade")
                    await self.send_telegram_message(
                        f"ℹ️ Sell skipped - balance {amount} below minimum {self.MIN_PI_ORDER} PI"
                    )
                    return None

            # Log the trade attempt
            logger.info(f"Attempting to {side} {amount} PI at {current_price} USDT")

            # Execute the trade
            order = self.exchange.create_order(
                symbol=self.symbol,
                type='market',
                side=side,
                amount=amount
            )

            # Rest of your execute_trade function...
            # Get fill details
            filled_order = self.exchange.fetch_order(order['id'], self.symbol)
            price = filled_order.get('average') or filled_order.get('price') or current_price

            # Calculate trade value and update stats
            total_value = float(price) * amount
            profit = total_value if side == "sell" else -total_value

            # Update daily stats
            self.daily_stats['trades'] += 1
            self.daily_stats['profit'] += profit
            self.daily_stats['best_trade'] = max(self.daily_stats['best_trade'], profit)
            self.daily_stats['worst_trade'] = min(self.daily_stats['worst_trade'], profit)

            # Get updated balances
            new_balances = await self.get_balances()

            # Select and format message
            if profit > 0:
                message = random.choice(self.profit_messages).format(profit=abs(profit))
            else:
                message = random.choice(self.loss_messages).format(loss=abs(profit))

            # Add enhanced trade details
            message += f"\n\n📊 Trade Details:\n" \
                       f"{'📈' if side == 'buy' else '📉'} {side.upper()}\n" \
                       f"💰 Amount: {amount:.2f} PI\n" \
                       f"💵 Price: ${price:.4f}\n" \
                       f"🎯 Signal Strength: {signal_strength}/3\n" \
                       f"🌊 Market Sentiment: {'Bullish 🐂' if sentiment > 0 else 'Bearish 🐻'}\n\n" \
                       f"💼 Updated Portfolio:\n" \
                       f"PI Balance: {new_balances['PI']:.2f}\n" \
                       f"USDT Balance: ${new_balances['USDT']:.2f}"

            await self.send_telegram_message(message)

            # Set trailing stop if buying
            if side == "buy":
                await self.set_trailing_stop(price, amount)

            return order
        except Exception as e:
            await self.send_telegram_message(f"⚠️ Trade Error: {str(e)}")
            return None

    async def set_trailing_stop(self, entry_price, amount):
        """Set and monitor trailing stop loss"""
        try:
            stop_price = entry_price * (1 - self.trailing_stop_pct)
            highest_price = entry_price

            while self.trading_allowed:
                ticker = self.exchange.fetch_ticker(self.symbol)
                current_price = ticker['last']

                if current_price > highest_price:
                    highest_price = current_price
                    stop_price = highest_price * (1 - self.trailing_stop_pct)

                if current_price <= stop_price:
                    # Execute stop loss
                    order = self.exchange.create_order(
                        symbol=self.symbol,
                        type='market',
                        side='sell',
                        amount=amount
                    )

                    message = (
                        f"🛑 Trailing Stop Triggered!\n"
                        f"Entry: ${entry_price:.4f}\n"
                        f"Exit: ${current_price:.4f}\n"
                        f"Protected Profit/Loss: ${(current_price - entry_price) * amount:.2f}"
                    )
                    await self.send_telegram_message(message, is_important=True)
                    break

                await asyncio.sleep(5)
        except Exception as e:
            logger.error(f"Error in trailing stop: {str(e)}")

    def calculate_signals(self, df):
        """Enhanced technical analysis"""
        try:
            # Standard indicators
            df['SMA7'] = df['close'].rolling(window=self.short_window).mean()
            df['SMA21'] = df['close'].rolling(window=self.long_window).mean()

            # Enhanced RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))

            # Volume analysis
            df['Volume_MA'] = df['volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['volume'] / df['Volume_MA']

            # Momentum
            df['ROC'] = df['close'].pct_change(periods=10) * 100

            # Volatility
            df['ATR'] = df['high'].rolling(window=14).max() - df['low'].rolling(window=14).min()

            return df
        except Exception as e:
            logger.error(f"Error calculating signals: {str(e)}")
            return None

    async def check_loss_threshold(self):
        """Check if losses exceed threshold"""
        try:
            current_balance = (await self.get_balances())['PI']

            # Ensure we have valid numbers before comparison
            if self.initial_balance is None:
                self.initial_balance = 0.0

            if current_balance is None:
                current_balance = 0.0

            # Convert to float to ensure proper comparison
            initial_balance = float(self.initial_balance)
            current_balance = float(current_balance)

            self.current_loss = initial_balance - current_balance

            logger.info(
                f"Loss check - Initial: {initial_balance}, Current: {current_balance}, Loss: {self.current_loss}")

            if self.current_loss >= self.loss_threshold:
                self.trading_allowed = False
                message = (
                    f"⚠️ Loss threshold reached!\n"
                    f"Initial balance: {initial_balance:.4f} PI\n"
                    f"Current balance: {current_balance:.4f} PI\n"
                    f"Total loss: {self.current_loss:.4f} PI\n"
                    f"Trading halted. Send /resume to continue trading."
                )
                await self.send_telegram_message(message)
        except Exception as e:
            logger.error(f"Error checking loss threshold: {str(e)}")
            # Don't halt trading on error
            self.current_loss = 0

    async def run(self):
        while self.running:
            try:
                balances = await self.get_balances()
                logger.info(f"Current balances - PI: {balances['PI']}, USDT: {balances['USDT']}")

                df = self.get_market_data()
                if df is None:
                    continue

                df = self.calculate_signals(df)
                if df is None:
                    continue

                # Check buy conditions FIRST if we have low PI balance
                if balances['PI'] < self.MIN_PI_ORDER:
                    buy_signals = self.should_buy(df)
                    if buy_signals and balances['USDT'] > (self.min_usdt_reserve + self.MIN_USDT_ORDER):
                        logger.info("Low PI balance - attempting to buy")
                        await self.execute_trade('buy', buy_signals)
                        await asyncio.sleep(60)  # Wait a bit after buying
                        continue

                # Only check sell conditions if we have enough PI to sell
                if balances['PI'] >= self.MIN_PI_ORDER:
                    sell_signals = self.should_sell(df)
                    if sell_signals:
                        await self.execute_trade('sell', sell_signals)

                await self.check_loss_threshold()
                await asyncio.sleep(60)

            except Exception as e:
                logger.error(f"Error in main loop: {str(e)}")
                await asyncio.sleep(60)


async def handle_resume(update, context):
    """Handle /resume command"""
    bot = context.bot_data['trading_bot']
    bot.trading_allowed = True
    await bot.initialize()
    await update.message.reply_text("Trading resumed! 🚀")


async def main():
    """Main function"""
    trading_bot = None
    try:
        # Initialize the bot
        trading_bot = TradingBot()

        # Initialize the Telegram application
        app = Application.builder().token(os.getenv('TELEGRAM_TOKEN')).build()
        trading_bot.app = app

        # Store the trading bot instance
        app.bot_data['trading_bot'] = trading_bot

        # Add command handlers
        app.add_handler(CommandHandler("resume", handle_resume))

        # Initialize bot task
        bot_task = asyncio.create_task(trading_bot.run())

        # Start polling in the background
        await app.initialize()
        await app.start()
        await app.updater.start_polling()

        # Wait for the bot task or interruption
        try:
            await bot_task
        except asyncio.CancelledError:
            pass
        finally:
            # Cleanup
            await trading_bot.stop()
            await app.updater.stop()
            await app.stop()
            await app.shutdown()

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        if trading_bot:
            try:
                await trading_bot.stop()
                if trading_bot.app:
                    await trading_bot.app.shutdown()
            except Exception as cleanup_error:
                logger.error(f"Error during cleanup: {str(cleanup_error)}")


if __name__ == "__main__":
    # Handle graceful shutdown
    loop = asyncio.get_event_loop()


    def handle_shutdown(sig, frame):
        loop.stop()
        logger.info("Received shutdown signal...")


    # Register signal handlers
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    try:
        loop.run_until_complete(main())
    finally:
        loop.close()
