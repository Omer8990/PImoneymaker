import os
import logging
from datetime import datetime
import asyncio
import signal
import ccxt
from telegram.ext import Application, CommandHandler
import pandas as pd
import numpy as np
from dotenv import load_dotenv

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
        self.trading_allowed = True
        self.running = True

        # Technical parameters
        self.short_window = 10
        self.long_window = 30
        self.rsi_period = 7
        self.rsi_oversold = 40
        self.rsi_overbought = 60

        self.min_trade_size = 5
        self.max_position_size = None
        self.initial_balance = None
        self.current_loss = 0

    async def get_pi_balance(self):
        """Get PI balance from exchange"""
        try:
            balance = self.exchange.fetch_balance()
            return float(balance.get('PI', {}).get('free', 0))
        except Exception as e:
            logger.error(f"Error fetching balance: {str(e)}")
            return 0

    async def send_telegram_message(self, message):
        """Send message via Telegram"""
        try:
            if self.app and self.chat_id:
                await self.app.bot.send_message(chat_id=self.chat_id, text=message)
        except Exception as e:
            logger.error(f"Error sending Telegram message: {str(e)}")

    async def initialize(self):
        """Initialize balance-dependent parameters"""
        self.initial_balance = await self.get_pi_balance()
        self.max_position_size = self.initial_balance * 0.25 if self.initial_balance else 0
        logger.info(f"Initialized with balance: {self.initial_balance} PI")

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
            df['SMA10'] = df['close'].rolling(window=self.short_window).mean()
            df['SMA30'] = df['close'].rolling(window=self.long_window).mean()

            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))

            # Volume indicators
            df['Volume_MA'] = df['volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['volume'] / df['Volume_MA']

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
                last_row['close'] > last_row['SMA30'],
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
                last_row['close'] < last_row['SMA30'],
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
        """Execute trade on exchange"""
        try:
            amount = self.calculate_position_size(signal_strength)
            order = self.exchange.create_order(
                symbol=self.symbol,
                type='market',
                side=side,
                amount=amount
            )

            # Fetch the actual order to get the filled price
            filled_order = self.exchange.fetch_order(order['id'], self.symbol)

            # Calculate average fill price
            if filled_order['average'] is not None:
                price = filled_order['average']
            elif filled_order['price'] is not None:
                price = filled_order['price']
            else:
                # Fallback to current market price
                ticker = self.exchange.fetch_ticker(self.symbol)
                price = ticker['last']

            emoji = "🚀" if side == "buy" else "💰"
            message = f"{emoji} Trade executed:\n" \
                      f"Side: {side.upper()}\n" \
                      f"Amount: {amount} PI\n" \
                      f"Price: {price}\n" \
                      f"Total: ${float(price) * amount:.2f}\n" \
                      f"Signal Strength: {signal_strength}/3"

            await self.send_telegram_message(message)
            return order
        except Exception as e:
            await self.send_telegram_message(f"Error executing trade: {str(e)}")
            return None

    async def check_loss_threshold(self):
        """Check if losses exceed threshold"""
        current_balance = await self.get_pi_balance()
        self.current_loss = self.initial_balance - current_balance

        if self.current_loss >= self.loss_threshold:
            self.trading_allowed = False
            message = f"⚠️ Loss threshold reached!\n" \
                      f"Initial balance: {self.initial_balance} PI\n" \
                      f"Current balance: {current_balance} PI\n" \
                      f"Total loss: {self.current_loss} PI\n" \
                      f"Trading halted. Send /resume to continue trading."
            await self.send_telegram_message(message)

    async def run(self):
        """Main trading loop"""
        await self.initialize()
        logger.info("Trading bot started")
        await self.send_telegram_message("🤖 Trading bot is now running!")

        while self.running:
            try:
                if not self.trading_allowed:
                    await asyncio.sleep(60)
                    continue

                df = self.get_market_data()
                if df is None:
                    continue

                df = self.calculate_signals(df)
                if df is None:
                    continue

                buy_signals = sum([self.should_buy(df) for _ in range(3)])
                sell_signals = sum([self.should_sell(df) for _ in range(3)])

                current_balance = await self.get_pi_balance()

                if buy_signals >= 2 and current_balance < self.initial_balance:
                    await self.execute_trade('buy', buy_signals)
                elif sell_signals >= 2 and current_balance > 0:
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
