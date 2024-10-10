require('dotenv').config();
const axios = require('axios');
import {ChatOpenAI} from "@langchain/openai";
import {ChatPromptTemplate} from '@langchain/core/prompts';
import {StructuredOutputParser} from '@langchain/core/output_parsers';
import {z} from 'zod';

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
    temperature: 0,
});

const zodSchema = z.object({
    answer: z.string().describe("Answer to the user's question"),
    outlook: z
        .enum(['positive', 'negative', 'neutral'])
        .describe('Overall outlook for the stock price'),
    recommendation: z
        .string()
        .describe('Investment recommendation based on the analysis'),
});

const parser = StructuredOutputParser.fromZodSchema(zodSchema);
const formatInstructions = parser.getFormatInstructions();
const prompt = ChatPromptTemplate.fromTemplate(
    `You are a financial analyst assistant.
Provide an analysis of the stock price for {stockSymbol}.

{format_instructions}

Current stock price: {stockPrice}
`
);

const chain = prompt.pipe(model).pipe(parser);

(async () => {
    const stockSymbol = 'EPAM';
    const stockPrice = await getStockPrice(stockSymbol);
    const response = await chain.invoke({
        stockSymbol,
        stockPrice,
        format_instructions: formatInstructions,
    });

    console.log('Answer:', response.answer);
    console.log('Outlook:', response.outlook);
    console.log('Recommendation:', response.recommendation);

    if (response.outlook === 'positive') {
        console.log('Executing code for positive outlook...');
        // Execute some code for a positive outlook
    } else if (response.outlook === 'negative') {
        console.log('Executing code for negative outlook...');
        // Execute some code for a negative outlook
    } else {
        console.log('Executing code for neutral outlook...');
        // Execute some code for a neutral outlook
    }
})();
