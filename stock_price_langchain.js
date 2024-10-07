require('dotenv').config();
const axios = require('axios');
import {ChatOpenAI} from "@langchain/openai";
import {StringOutputParser} from "@langchain/core/output_parsers";
import {ChatPromptTemplate} from "@langchain/core/prompts";

const openaiApiKey = process.env.OPENAI_API_KEY;
const alphaVantageApiKey = process.env.ALPHA_VANTAGE_API_KEY;

async function getStockPrice(symbol) {
    const url = `https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=${symbol}&interval=5min&apikey=${alphaVantageApiKey}`;
    const response = await axios.get(url);
    const timeSeries = response.data['Time Series (5min)'];
    const latestTime = Object.keys(timeSeries)[0];
    return timeSeries[latestTime]['1. open'];
}

const model = new ChatOpenAI({
    openAIApiKey: openaiApiKey,
    modelName: 'gpt-4o-mini',
});

const prompt = ChatPromptTemplate.fromTemplate(
    'What does a stock price of {stockPrice} mean for {stockSymbol}?'
);

const chain = prompt.pipe(model).pipe(new StringOutputParser());

(async () => {
    const stockSymbol = 'EPAM';
    const stockPrice = await getStockPrice(stockSymbol);
    const response = await chain.invoke({stockPrice, stockSymbol});

    if (response.includes('positive outlook')) {
        // Execute some code for a positive outlook
        console.log("Positive!")
    } else if (response.includes('negative outlook')) {
        // Execute some code for a negative outlook
        console.log("Negative!")
    }
    console.log(response);
})();

