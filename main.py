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
        """Send formatted message via Telegram"""
        try:
            if self.app and self.chat_id:
                timestamp = datetime.now().strftime("%H:%M:%S")
                formatted_message = f"⏰ {timestamp}\n{message}"

                if is_important:
                    formatted_message = f"🔔 IMPORTANT UPDATE 🔔\n{formatted_message}"

                await self.app.bot.send_message(
                    chat_id=self.chat_id,
                    text=formatted_message,
                    parse_mode='HTML'
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

            if usdt_available <= 0:
                return 0

            # Calculate USDT to use for this trade
            usdt_to_use = min(
                usdt_available * self.usdt_position_pct,
                self.max_usdt_per_trade
            )

            # Calculate PI amount we can buy
            pi_amount = usdt_to_use / current_price

            # Round to 2 decimal places for better order accuracy
            return round(pi_amount, 2)
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
        balances = await self.get_balances()
        self.initial_balance = balances['PI']

        # Log initial balances
        logger.info(f"Initialized with balances: {balances['PI']} PI, {balances['USDT']} USDT")

        # Send initial balance message
        await self.send_telegram_message(
            f"🏦 Initial Balances:\n"
            f"PI: {balances['PI']:.2f}\n"
            f"USDT: ${balances['USDT']:.2f}\n\n"
            f"Ready to trade! 🚀"
        )

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
            conditions = [
                last_row['RSI'] < self.rsi_oversold,
                last_row['SMA7'] > last_row['SMA21'],  # Changed from SMA30
                last_row['Volume_Ratio'] > 1.2
            ]
            return sum(conditions) >= 2
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

            # Get current market price
            ticker = self.exchange.fetch_ticker(self.symbol)
            current_price = ticker['last']

            if side == 'buy':
                # Calculate amount based on available USDT
                amount = await self.calculate_buy_amount(current_price)
                if amount <= 0:
                    await self.send_telegram_message("⚠️ Not enough USDT available for trading!")
                    return None
            else:  # sell
                # Get available PI balance for selling
                balances = await self.get_balances()
                amount = min(
                    balances['PI'],
                    self.calculate_position_size(signal_strength * (1 + sentiment))
                )
                if amount <= 0:
                    await self.send_telegram_message("⚠️ No PI available to sell!")
                    return None

            # Execute the trade
            order = self.exchange.create_order(
                symbol=self.symbol,
                type='market',
                side=side,
                amount=amount
            )

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

            await self.send_telegram_message(message, is_important=True)

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
            self.current_loss = float(self.initial_balance) - float(current_balance)

            if self.current_loss >= self.loss_threshold:
                self.trading_allowed = False
                message = (
                    f"⚠️ Loss threshold reached!\n"
                    f"Initial balance: {self.initial_balance} PI\n"
                    f"Current balance: {current_balance} PI\n"
                    f"Total loss: {self.current_loss} PI\n"
                    f"Trading halted. Send /resume to continue trading."
                )
                await self.send_telegram_message(message)
        except Exception as e:
            logger.error(f"Error checking loss threshold: {str(e)}")

    async def run(self):
        """Main trading loop with error handling"""
        await self.initialize()
        await self.send_telegram_message("🤖 J.A.R.V.I.S Trading System Online! Ready to make some money, boss! 💰")

        last_daily_report = datetime.now()

        while self.running:
            try:
                # Send daily report
                if datetime.now() - last_daily_report > timedelta(days=1):
                    await self.send_daily_report()
                    last_daily_report = datetime.now()

                if not self.trading_allowed:
                    await asyncio.sleep(60)
                    continue

                df = self.get_market_data()
                if df is None:
                    continue

                df = self.calculate_signals(df)
                if df is None:
                    continue

                # Get current balances
                balances = await self.get_balances()

                # Check buy conditions
                buy_signals = self.should_buy(df)
                if buy_signals and balances['USDT'] > self.min_usdt_reserve:
                    await self.execute_trade('buy', buy_signals)

                # Check sell conditions
                sell_signals = self.should_sell(df)
                if sell_signals and balances['PI'] > 0:
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
