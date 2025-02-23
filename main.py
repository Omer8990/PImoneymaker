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
        self.usdt_position_pct = 0.5  # Currently uses 50% of available USDT
        self.MIN_PI_ORDER = 1.0  # Set to 1 PI minimum
        self.MIN_USDT_ORDER = 5.0  # Set to 5 USDT minimum

        # Also increase the minimum USDT reserve to ensure we have enough for fees
        self.min_usdt_reserve = 15  # Increased from 10

        # Adjust USDT position percentage to ensure larger orders
        self.usdt_position_pct = 0.5  # Increased from 0.2 to use 50% of available USDT

        # Personality messages
        self.profit_messages = [
            "Sir, your investment acumen proves formidable once again. Profit secured: {profit:.2f} USDT. 💰",
            "Excellent play, sir. We've locked in {profit:.2f} USDT. Shall I prepare the celebratory protocols? 🎯",
            "Mr. Stark would be proud - another successful trade at {profit:.2f} USDT. Your market intuition remains unmatched. ⚡",
            "Profit secured: {profit:.2f} USDT. I must say, sir, your trading algorithm continues to impress. 🎨"
        ]

        self.loss_messages = [
            "A minor setback of {loss:.2f} USDT, sir. But as you often say, we learn more from our failures than our successes. 📊",
            "Sir, we've encountered a temporary deviation of {loss:.2f} USDT. Recalibrating strategies as we speak. 🔄",
            "Down {loss:.2f} USDT. Nothing to worry about, sir - I've seen you pull through worse scenarios. Adjusting parameters. 🛠️",
            "A slight miscalculation: -{loss:.2f} USDT. Shall I run a diagnostic on our trading parameters? 🔍"
        ]

        self.initialization_messages = [
            "Systems online, sir. Initial diagnostics complete:\n"
            "PI: {pi:.4f}\n"
            "USDT: ${usdt:.4f}\n\n"
            "Trading protocols activated and ready for your command. 🚀",

            "All systems operational, sir:\n"
            "PI Balance: {pi:.4f}\n"
            "USDT Reserve: ${usdt:.4f}\n\n"
            "The market awaits your expertise. 💫"
        ]

        self.daily_report_template = (
            "📊 Daily Performance Analytics 📊\n\n"
            "Sir, here's your daily trading summary:\n\n"
            "Transactions Executed: {trades} 🎯\n"
            "Net Profit/Loss: {profit:.2f} USDT\n"
            "Peak Performance: {best:.2f} USDT 🏆\n"
            "Learning Opportunity: {worst:.2f} USDT 📉\n\n"
            "Current Portfolio Status:\n"
            "PI: {pi:.2f}\n"
            "USDT: ${usdt:.2f}\n"
            "Initial PI Holdings: {initial:.2f}\n\n"
            "{performance_comment}"
        )

        self.performance_comments = {
            'positive': [
                "Excellent day, sir. The algorithms performed admirably. 🌟",
                "Another successful day in the books. Your strategy proves effective once again. 🚀",
                "The numbers speak for themselves, sir. Shall I prepare for tomorrow's opportunities? 💫"
            ],
            'negative': [
                "A challenging day, sir, but nothing we haven't handled before. Recalibrating for tomorrow. 💪",
                "Today's market proved... interesting. But as you say, sir - we learn and adapt. 🔄",
                "Perhaps a slight adjustment to our parameters might be in order for tomorrow, sir? 🛠️"
            ]
        }

        self.trade_details_template = (
            "\n\n📊 Trade Analysis:\n"
            "{direction} {side.upper()} (Position Size: {position_pct:.0f}%)\n"
            "Amount: {amount:.2f} PI\n"
            "Market Rate: ${price:.4f}\n"
            "Confidence Index: {signal_strength}/3\n"
            "Market Sentiment: {sentiment}\n\n"
            "Updated Holdings:\n"
            "PI: {pi_balance:.2f}\n"
            "USDT: ${usdt_balance:.2f}"
        )

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
        """Enhanced daily report with JARVIS-style messaging"""
        try:
            balances = await self.get_balances()

            # Determine performance comment
            comment_list = (self.performance_comments['positive']
                            if self.daily_stats['profit'] > 0
                            else self.performance_comments['negative'])
            performance_comment = random.choice(comment_list)

            report = self.daily_report_template.format(
                trades=self.daily_stats['trades'],
                profit=self.daily_stats['profit'],
                best=self.daily_stats['best_trade'],
                worst=self.daily_stats['worst_trade'],
                pi=balances['PI'],
                usdt=balances['USDT'],
                initial=self.initial_balance,
                performance_comment=performance_comment
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
        """Initialize with JARVIS-style welcome message"""
        try:
            balances = await self.get_balances()
            if balances['PI'] is not None:
                self.initial_balance = float(balances['PI'])
            else:
                self.initial_balance = 0.0

            welcome_message = random.choice(self.initialization_messages).format(
                pi=balances['PI'],
                usdt=balances['USDT']
            )

            await self.send_telegram_message(welcome_message)
        except Exception as e:
            logger.error(f"Error in initialization: {str(e)}")
            self.initial_balance = 0.0

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

    def should_buy(self, df):
        last_row = df.iloc[-1]

        # Add more sophisticated conditions
        conditions = [
            last_row['RSI'] < 30,  # More conservative
            last_row['SMA7'] > last_row['SMA21'],
            last_row['Volume_Ratio'] > 1.5,  # Higher volume requirement
            last_row['close'] < last_row['SMA21'],  # Below moving average
            last_row['ROC'] < -2  # Rate of change showing downward momentum
        ]

        # Require more conditions to be met
        return sum(conditions) >= 3

    def should_sell(self, df):
        last_row = df.iloc[-1]
        conditions = [
            last_row['RSI'] > 70,  # More aggressive profit taking
            last_row['SMA7'] < last_row['SMA21'],
            last_row['Volume_Ratio'] > 1.2,  # Good volume for selling
            last_row['ROC'] > 3  # Strong upward momentum
        ]
        return sum(conditions) >= 2

    def calculate_position_size(self, signal_strength):
        """Calculate position size based on signal strength"""
        base_size = self.min_trade_size
        if self.max_position_size is None:
            return base_size
        return min(base_size * signal_strength, self.max_position_size)

    async def execute_trade(self, side, signal_strength):
        """Execute trade with enhanced USDT management and partial position taking"""
        try:
            sentiment = await self.get_market_sentiment()
            ticker = self.exchange.fetch_ticker(self.symbol)
            current_price = ticker['last']

            # Determine position size based on signal strength
            position_sizes = {
                3: 0.8,  # Strong signal: 80% of available balance
                2: 0.5,  # Medium signal: 50% of available balance
                1: 0.3  # Weak signal: 30% of available balance
            }
            position_percent = position_sizes.get(signal_strength, 0.3)  # Default to 30% if invalid strength

            if side == 'buy':
                # Calculate buy amount with position sizing
                full_amount = await self.calculate_buy_amount(current_price)
                amount = full_amount * position_percent

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
                full_amount = balances['PI']
                amount = full_amount * position_percent

                if amount < self.MIN_PI_ORDER:
                    logger.info(f"Sell amount {amount} is below minimum {self.MIN_PI_ORDER}, skipping trade")
                    await self.send_telegram_message(
                        f"ℹ️ Sell skipped - amount {amount} below minimum {self.MIN_PI_ORDER} PI"
                    )
                    return None

            # Log the trade attempt
            logger.info(
                f"Attempting to {side} {amount:.2f} PI ({position_percent * 100:.0f}% position) at {current_price} USDT")

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

            # Add enhanced trade details with position size information
            message += f"\n\n📊 Trade Details:\n" \
                       f"{'📈' if side == 'buy' else '📉'} {side.upper()} ({position_percent * 100:.0f}% position)\n" \
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
