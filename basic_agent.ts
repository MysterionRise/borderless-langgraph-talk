require('dotenv').config();

import {tool} from '@langchain/core/tools';
import {z} from 'zod';
import axios from 'axios';
import {ToolNode} from '@langchain/langgraph/prebuilt';
import {
    StateGraph,
    MessagesAnnotation,
    END,
    START,
} from '@langchain/langgraph';
import {ChatOpenAI} from '@langchain/openai';
import {HumanMessage} from '@langchain/core/messages';

async function getStockPrice(symbol) {
    const url = `https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=${symbol}&apikey=${process.env.ALPHA_VANTAGE_API_KEY}`;
    const response = await axios.get(url);
    const price = response.data['Global Quote']['05. price'];
    return parseFloat(price);
}

const getStockPriceTool = tool(
    async (input) => {
        const stockPrice = await getStockPrice(input.symbol);
        return `The current price of ${input.symbol} is $${stockPrice}.`;
    },
    {
        name: 'get_stock_price',
        description: 'Fetches the current stock price for a given symbol.',
        schema: z.object({
            symbol: z.string().describe('Stock symbol to get the price for.'),
        }),
    }
);

const tools = [getStockPriceTool];
const toolNode = new ToolNode(tools);

const model = new ChatOpenAI({temperature: 0}).bindTools(tools);

const shouldContinue = (state) => {
    const {messages} = state;
    const lastMessage = messages[messages.length - 1];
    if (
        'tool_calls' in lastMessage &&
        Array.isArray(lastMessage.tool_calls) &&
        lastMessage.tool_calls.length
    ) {
        return 'tools';
    }
    return END;
};

const callModel = async (state) => {
    const {messages} = state;
    const response = await model.invoke(messages);
    return {messages: [response]};
};

const workflow = new StateGraph(MessagesAnnotation)
    .addNode('agent', callModel)
    .addNode('tools', toolNode)
    .addEdge(START, 'agent')
    .addConditionalEdges('agent', shouldContinue)
    .addEdge('tools', 'agent');

const app = workflow.compile();

(async () => {
    const stream = await app.stream(
        {
            messages: [new HumanMessage('Should I buy EPAM stock?')],
        },
        {
            streamMode: 'values',
        }
    );

    for await (const chunk of stream) {
        const lastMessage = chunk.messages[chunk.messages.length - 1];
        const type = lastMessage._getType();
        const content = lastMessage.content;
        const toolCalls = lastMessage.tool_calls;
        console.dir(
            {
                type,
                content,
                toolCalls,
            },
            {depth: null}
        );
    }
})();
