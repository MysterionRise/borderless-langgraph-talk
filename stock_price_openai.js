require('dotenv').config();
const axios = require('axios');
const OpenAI = require('openai');
const openai = new OpenAI();

const openaiApiKey = process.env.OPENAI_API_KEY;
const alphaVantageApiKey = process.env.ALPHA_VANTAGE_API_KEY;

async function getStockPrice(symbol) {
  const url = `https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=${symbol}&interval=5min&apikey=${alphaVantageApiKey}`;
  const response = await axios.get(url);
  const timeSeries = response.data['Time Series (5min)'];
  const latestTime = Object.keys(timeSeries)[0];
  return timeSeries[latestTime]['1. open'];
}

async function askOpenAI(stockSymbol, stockPrice) {
  const completion = await openai.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [
      { role: "system", content: "You are a helpful assistant." },
      {
        role: "user",
        content: `What does a stock price of ${stockPrice} mean for ${stockSymbol}?`,
      },
    ],
  });

  return completion.choices[0].message.content;
}

(async () => {
  const stockSymbol = 'AAPL';
  const stockPrice = await getStockPrice(stockSymbol);
  const aiResponse = await askOpenAI(stockSymbol, stockPrice);
  // WHAT IF WE WANT TO DO SOMETHING BASED ON THIS LLM RESPONSE
  console.log(aiResponse);
})();
