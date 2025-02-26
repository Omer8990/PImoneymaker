import os
import logging
import json
from datetime import datetime
import asyncio
import ccxt
from telegram.ext import Application, CommandHandler
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

# Initialize OKX exchange
exchange = ccxt.okx({
    'apiKey': os.getenv('OKX_API_KEY'),
    'secret': os.getenv('OKX_SECRET'),
    'password': os.getenv('OKX_PASSWORD'),
    'enableRateLimit': True,
})

# Telegram configuration
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# Trading parameters (high risk settings)
SYMBOL = 'PI/USDT'
BASE_ORDER_SIZE = 0.85  # Use 85% of available USDT for each trade
PROFIT_THRESHOLD = 0.03  # 3% profit target
STOP_LOSS = 0.05  # 5% stop loss
VOLATILITY_THRESHOLD = 0.02  # 2% price movement to trigger analysis
CHECK_INTERVAL = 60  # Check market every 60 seconds

# Cooldown timer between trades to prevent overtrading
TRADE_COOLDOWN = 300  # 5 minutes

# Jarvis persona responses
JARVIS_GREETINGS = [
    "Good day, sir. JARVIS online and monitoring PI crypto markets.",
    "At your service, sir. PI market surveillance initialized.",
    "Booting up crypto trading protocols. Ready when you are, sir.",
    "JARVIS active. PI market analysis systems online."
]

JARVIS_BUY_MESSAGES = [
    "Sir, I've detected a favorable entry point for PI. Executing purchase protocol.",
    "Market conditions for PI appear promising. Buying now, sir.",
    "Opportunity detected. Acquiring PI at what appears to be a discounted rate.",
    "Deploying capital into PI. The risk-reward ratio looks particularly enticing."
]

JARVIS_SELL_MESSAGES = [
    "Sir, profit target achieved. Executing sell order for PI holdings.",
    "PI position has reached optimal exit point. Selling now.",
    "The PI rocket has reached our desired altitude. Parachuting out.",
    "Profit secured, sir. Would you like me to prepare a celebratory beverage?"
]

JARVIS_STOP_LOSS_MESSAGES = [
    "Stop loss triggered, sir. Cutting our losses on PI as instructed.",
    "PI is underperforming expectations. Executing stop loss protocol.",
    "Sometimes you win, sometimes you learn, sir. Exiting PI position.",
    "Strategic retreat initiated. PI position liquidated to preserve capital."
]

JARVIS_ANALYSIS_MESSAGES = [
    "Analyzing PI market patterns. The volatility reminds me of your heart rate during test flights.",
    "Running technical analysis on PI. These market structures are quite fascinating.",
    "Market conditions for PI are changing rapidly. I'm monitoring closely.",
    "PI market assessment in progress. The algorithms are detecting interesting patterns."
]

last_trade_time = None
in_position = False
entry_price = 0


async def send_telegram_message(message):
    """Send message to Telegram"""
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    await application.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)


async def get_market_data():
    """Get current market data for PI/USDT"""
    try:
        ticker = exchange.fetch_ticker(SYMBOL)
        return {
            'price': ticker['last'],
            'volume': ticker['quoteVolume'],
            'change_24h': ticker['percentage'],
            'high': ticker['high'],
            'low': ticker['low']
        }
    except Exception as e:
        logger.error(f"Error fetching market data: {e}")
        await send_telegram_message(f"Sir, we have a problem accessing market data: {e}")
        return None


async def get_available_balance():
    """Get available USDT balance"""
    try:
        balance = exchange.fetch_balance()
        return balance['USDT']['free']
    except Exception as e:
        logger.error(f"Error fetching balance: {e}")
        await send_telegram_message(f"Unable to access wallet funds, sir. Error: {e}")
        return 0


async def analyze_market():
    """Analyze market conditions for trading opportunities"""
    market_data = await get_market_data()
    if not market_data:
        return False, 0

    # Get recent candles for analysis
    try:
        candles = exchange.fetch_ohlcv(SYMBOL, '5m', limit=20)

        # Calculate volatility (standard deviation of returns)
        closes = [candle[4] for candle in candles]
        returns = [closes[i] / closes[i - 1] - 1 for i in range(1, len(closes))]
        volatility = sum([abs(r) for r in returns]) / len(returns)

        # Calculate RSI
        gains = sum([r for r in returns if r > 0])
        losses = sum([abs(r) for r in returns if r < 0])

        if losses == 0:
            rsi = 100
        else:
            rs = gains / losses
            rsi = 100 - (100 / (1 + rs))

        # Simple moving averages
        sma_short = sum(closes[-5:]) / 5
        sma_long = sum(closes[-15:]) / 15

        # Calculate risk score (higher = more favorable for buying)
        risk_score = 0

        # Trend analysis
        if sma_short > sma_long:
            risk_score += 3

        # Volatility analysis
        if volatility > VOLATILITY_THRESHOLD:
            risk_score += 2

        # RSI analysis
        if rsi < 30:  # Oversold
            risk_score += 3
        elif rsi > 70:  # Overbought
            risk_score -= 3

        # Recent performance
        if market_data['change_24h'] < -5:  # Big dip
            risk_score += 2

        # Random factor for high risk (20% chance of buying regardless)
        if random.random() < 0.2:
            risk_score += 4

        logger.info(f"Analysis complete - Risk score: {risk_score}, RSI: {rsi}, Volatility: {volatility}")

        # Decision threshold
        buy_signal = risk_score >= 4
        return buy_signal, market_data['price']

    except Exception as e:
        logger.error(f"Error in market analysis: {e}")
        await send_telegram_message(f"Sir, my analysis circuits encountered an error: {e}")
        return False, 0


async def buy_pi(current_price):
    """Execute buy order for PI"""
    global in_position, entry_price, last_trade_time

    try:
        usdt_balance = await get_available_balance()
        order_size_usdt = usdt_balance * BASE_ORDER_SIZE

        if order_size_usdt < 10:  # Minimum trade size
            await send_telegram_message("Sir, available USDT balance is too low for meaningful trade execution.")
            return False

        # Calculate amount of PI to buy
        amount = order_size_usdt / current_price

        # Place market buy order
        order = exchange.create_market_buy_order(SYMBOL, amount)

        # Update position status
        in_position = True
        entry_price = current_price
        last_trade_time = datetime.now()

        # Send notification
        message = f"{random.choice(JARVIS_BUY_MESSAGES)}\n\n" \
                  f"📊 Trade Summary:\n" \
                  f"🔹 Bought: {amount:.4f} PI @ ${current_price:.4f}\n" \
                  f"🔹 Total: ${order_size_usdt:.2f} USDT\n" \
                  f"🔹 Target: ${current_price * (1 + PROFIT_THRESHOLD):.4f}\n" \
                  f"🔹 Stop Loss: ${current_price * (1 - STOP_LOSS):.4f}\n\n" \
                  f"Risk assessment: High. But as you say, sir, 'No risk, no reward.'"

        await send_telegram_message(message)
        logger.info(f"Buy order executed: {order}")
        return True

    except Exception as e:
        logger.error(f"Error placing buy order: {e}")
        await send_telegram_message(f"Sir, the buy order failed to execute: {e}")
        return False


async def sell_pi(current_price, reason="profit"):
    """Execute sell order for PI"""
    global in_position, entry_price, last_trade_time

    try:
        # Get current PI balance
        balance = exchange.fetch_balance()
        pi_amount = balance['PI']['free']

        if pi_amount * current_price < 10:  # Minimum trade value
            await send_telegram_message("PI position too small to sell, sir.")
            return False

        # Place market sell order
        order = exchange.create_market_sell_order(SYMBOL, pi_amount)

        # Calculate profit/loss
        entry_value = pi_amount * entry_price
        exit_value = pi_amount * current_price
        profit_loss = exit_value - entry_value
        profit_percent = (profit_loss / entry_value) * 100

        # Update position status
        in_position = False
        last_trade_time = datetime.now()

        # Select appropriate message based on reason
        if reason == "profit":
            message_templates = JARVIS_SELL_MESSAGES
        else:  # stop loss
            message_templates = JARVIS_STOP_LOSS_MESSAGES

        # Send notification
        message = f"{random.choice(message_templates)}\n\n" \
                  f"📊 Trade Summary:\n" \
                  f"🔹 Sold: {pi_amount:.4f} PI @ ${current_price:.4f}\n" \
                  f"🔹 Entry Price: ${entry_price:.4f}\n" \
                  f"🔹 P/L: ${profit_loss:.2f} ({profit_percent:.2f}%)\n\n"

        if profit_loss > 0:
            message += "A successful venture, sir. Perhaps this calls for a celebration."
        else:
            message += "Not all experimental trades succeed, sir. Recalibrating strategy."

        await send_telegram_message(message)
        logger.info(f"Sell order executed: {order}")
        return True

    except Exception as e:
        logger.error(f"Error placing sell order: {e}")
        await send_telegram_message(f"Sir, the sell order failed to execute: {e}")
        return False


async def check_exit_conditions(current_price):
    """Check if we should exit current position"""
    if not in_position:
        return

    profit_percentage = (current_price - entry_price) / entry_price

    # Check profit target
    if profit_percentage >= PROFIT_THRESHOLD:
        logger.info(f"Profit target hit: {profit_percentage:.2%}")
        await sell_pi(current_price, "profit")

    # Check stop loss
    elif profit_percentage <= -STOP_LOSS:
        logger.info(f"Stop loss hit: {profit_percentage:.2%}")
        await sell_pi(current_price, "stop_loss")


async def trading_loop():
    """Main trading loop"""
    global last_trade_time

    # Send startup message
    await send_telegram_message(
        f"{random.choice(JARVIS_GREETINGS)}\n\nMonitoring {SYMBOL} for high-risk opportunities. Safety protocols minimized as requested, sir.")

    while True:
        try:
            current_data = await get_market_data()
            if not current_data:
                await asyncio.sleep(CHECK_INTERVAL)
                continue

            current_price = current_data['price']

            # First check if we need to exit current position
            if in_position:
                await check_exit_conditions(current_price)

            # Then check if we should enter a new position
            else:
                # Ensure cooldown period has passed
                if last_trade_time and (datetime.now() - last_trade_time).seconds < TRADE_COOLDOWN:
                    await asyncio.sleep(CHECK_INTERVAL)
                    continue

                # Send occasional analysis message
                if random.random() < 0.3:  # 30% chance of sending analysis message
                    await send_telegram_message(random.choice(JARVIS_ANALYSIS_MESSAGES))

                # Analyze market for buy signal
                buy_signal, price = await analyze_market()
                if buy_signal:
                    await buy_pi(current_price)

            # Sleep before next check
            await asyncio.sleep(CHECK_INTERVAL)

        except Exception as e:
            logger.error(f"Error in trading loop: {e}")
            await send_telegram_message(f"Sir, I've encountered an unexpected error: {e}. Trading systems rebooting.")
            await asyncio.sleep(CHECK_INTERVAL * 2)


# Command handlers for manual control
async def start_command(update, context):
    """Start the bot"""
    await update.message.reply_text(
        "JARVIS trading protocol activated, sir. Monitoring PI/USDT for aggressive trading opportunities."
    )


async def status_command(update, context):
    """Show current status"""
    market_data = await get_market_data()
    balance = await get_available_balance()

    status_message = f"📊 Status Report - {datetime.now().strftime('%H:%M:%S')}\n\n"
    status_message += f"🔹 PI Price: ${market_data['price']:.4f}\n"
    status_message += f"🔹 24h Change: {market_data['change_24h']:.2f}%\n"
    status_message += f"🔹 USDT Balance: ${balance:.2f}\n"

    if in_position:
        profit_percentage = (market_data['price'] - entry_price) / entry_price * 100
        status_message += f"🔹 Position: LONG PI @ ${entry_price:.4f}\n"
        status_message += f"🔹 Current P/L: {profit_percentage:.2f}%\n"
        status_message += f"🔹 Target Exit: ${entry_price * (1 + PROFIT_THRESHOLD):.4f}\n"
        status_message += f"🔹 Stop Loss: ${entry_price * (1 - STOP_LOSS):.4f}\n"
    else:
        status_message += "🔹 Position: No active position\n"

    await update.message.reply_text(status_message)


async def buy_command(update, context):
    """Manual buy command"""
    global in_position

    if in_position:
        await update.message.reply_text("Already in position, sir. Perhaps selling first would be prudent?")
        return

    market_data = await get_market_data()
    await update.message.reply_text("Manual override accepted. Executing buy order, sir.")
    await buy_pi(market_data['price'])


async def sell_command(update, context):
    """Manual sell command"""
    global in_position

    if not in_position:
        await update.message.reply_text("No position to sell, sir. Perhaps we should acquire some PI first?")
        return

    market_data = await get_market_data()
    await update.message.reply_text("Manual sell protocol initiated. Liquidating PI position, sir.")
    await sell_pi(market_data['price'], "manual")


async def set_params_command(update, context):
    """Set trading parameters"""
    try:
        args = context.args

        if len(args) != 3:
            await update.message.reply_text(
                "Incorrect parameters. Format: /set_params [order_size] [profit] [stop_loss]\n"
                "Example: /set_params 0.85 0.03 0.05"
            )
            return

        global BASE_ORDER_SIZE, PROFIT_THRESHOLD, STOP_LOSS
        BASE_ORDER_SIZE = float(args[0])
        PROFIT_THRESHOLD = float(args[1])
        STOP_LOSS = float(args[2])

        await update.message.reply_text(
            f"Parameters updated successfully, sir:\n"
            f"🔹 Order Size: {BASE_ORDER_SIZE * 100}% of USDT\n"
            f"🔹 Profit Target: {PROFIT_THRESHOLD * 100}%\n"
            f"🔹 Stop Loss: {STOP_LOSS * 100}%"
        )
    except Exception as e:
        await update.message.reply_text(f"Error updating parameters: {e}")


async def setup_telegram_commands():
    """Set up Telegram command handlers"""
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # Add command handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("buy", buy_command))
    application.add_handler(CommandHandler("sell", sell_command))
    application.add_handler(CommandHandler("set_params", set_params_command))

    # Start the bot
    await application.initialize()
    await application.start()
    await application.updater.start_polling()


# Main function
async def main():
    """Run the bot"""
    # Setup logging
    logging.info("Starting PI trading bot")

    # Run Telegram bot and trading loop concurrently
    await asyncio.gather(
        setup_telegram_commands(),
        trading_loop()
    )


if __name__ == "__main__":
    # Run event loop
    asyncio.run(main())
