import {END, Annotation} from "@langchain/langgraph";
import {BaseMessage, HumanMessage, SystemMessage} from "@langchain/core/messages";
import {ChatOpenAI} from "@langchain/openai";
import {ChatPromptTemplate, MessagesPlaceholder} from "@langchain/core/prompts";
import {RunnableConfig} from "@langchain/core/runnables";
import {START, StateGraph} from "@langchain/langgraph";
import {createReactAgent} from "@langchain/langgraph/prebuilt";
import {z} from "zod";
import {JsonOutputToolsParser} from "@langchain/core/output_parsers/openai_tools";
import axios from 'axios';
import {DynamicStructuredTool} from "@langchain/core/tools";
import * as dotenv from 'dotenv';

dotenv.config();

const openaiApiKey = process.env.OPENAI_API_KEY;
const alphaVantageApiKey = process.env.ALPHA_VANTAGE_API_KEY;

const AgentState = Annotation.Root({
    messages: Annotation<BaseMessage[]>({
        reducer: (x, y) => x.concat(y),
        default: () => [],
    }),
    memory: Annotation<any>({
        reducer: (x, y) => ({...x, ...y}),
        default: () => ({}),
    }),
    next: Annotation<string>({
        reducer: (x, y) => y ?? x ?? END,
        default: () => END,
    }),
});

const stockPriceTool = new DynamicStructuredTool({
    name: "get_stock_price",
    description: "Fetches the latest stock price for a given stock symbol.",
    schema: z.object({
        symbol: z.string().describe("Stock symbol to fetch price for"),
    }),
    func: async ({symbol}) => {
        const alphaVantageApiKey = process.env.ALPHA_VANTAGE_API_KEY;
        const url = `https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=${symbol}&apikey=${alphaVantageApiKey}`;
        const response = await axios.get(url);
        const stockPrice = response.data['Global Quote']['05. price'];
        if (!stockPrice) {
            throw new Error(`Could not fetch stock price for symbol: ${symbol}`);
        }
        return stockPrice;
    },
});

const members = ["data_fetcher", "analyst", "reporter"] as const;

const systemPrompt = `
You are a supervisor tasked with managing a conversation between the following workers: {members}.
Given the following user request and the conversation so far, respond with the worker to act next.
Each worker will perform a task and respond with their results and status.
When finished, respond with FINISH.
`;

const options = [END, ...members];

const routingTool = {
    name: "route",
    description: "Select the next role.",
    schema: z.object({
        next: z.enum(options),
    }),
};

const prompt = ChatPromptTemplate.fromMessages([
    ["system", systemPrompt],
    new MessagesPlaceholder("messages"),
    [
        "system",
        "Given the conversation above, who should act next? Or should we FINISH? Select one of: {options}",
    ],
]);

const formattedPrompt = await prompt.partial({
    options: options.join(", "),
    members: members.join(", "),
});

const llm = new ChatOpenAI({
    openAIApiKey: openaiApiKey,
    modelName: 'gpt-4o-mini',
    temperature: 0,
});

const supervisorChain = formattedPrompt
    .pipe(llm.bindTools(
        [routingTool],
        {
            tool_choice: "route",
        },
    ))
    .pipe(new JsonOutputToolsParser())
    .pipe((x) => (x[0].args));

const supervisorNode = async (
    state: typeof AgentState.State,
    config?: RunnableConfig,
) => {
    const result = await supervisorChain.invoke(state, config);
    return {
        next: result.next,
    };
};

const dataFetcherAgent = createReactAgent({
    llm,
    tools: [stockPriceTool],
    messageModifier: new SystemMessage(
        "You are a data fetcher. Use the get_stock_price tool to fetch the latest stock price for the requested stock symbol."),
});

const dataFetcherNode = async (
    state: typeof AgentState.State,
    config?: RunnableConfig,
) => {
    const result = await dataFetcherAgent.invoke(state, config);
    const lastMessage = result.messages[result.messages.length - 1];

    const stockSymbolMatch = lastMessage.content.match(/stock price for (\w+): (\d+\.\d+)/);
    let stockSymbol = '';
    let stockPrice = '';

    if (stockSymbolMatch) {
        stockSymbol = stockSymbolMatch[1];
        stockPrice = stockSymbolMatch[2];
    }

    return {
        messages: [
            new HumanMessage({content: lastMessage.content, name: "DataFetcher"}),
        ],
        memory: {
            stockSymbol,
            stockPrice,
        },
    };
};

const analystAgent = createReactAgent({
    llm,
    tools: [],
    messageModifier: new SystemMessage("You are a financial analyst. Analyze the stock price provided in the conversation and determine the outlook (positive, negative, neutral) and make a recommendation."),
});

const analystNode = async (
    state: typeof AgentState.State,
    config?: RunnableConfig,
) => {
    const result = await analystAgent.invoke(state, config);
    const lastMessage = result.messages[result.messages.length - 1];

    return {
        messages: [
            new HumanMessage({content: lastMessage.content, name: "Analyst"}),
        ],
        memory: {
            analysis: lastMessage.content,
        },
    };
};

const reporterAgent = createReactAgent({
    llm,
    tools: [],
    messageModifier: new SystemMessage("You are the reporter. Summarize the analysis and provide it to the user."),
});

const reporterNode = async (
    state: typeof AgentState.State,
    config?: RunnableConfig,
) => {
    const result = await reporterAgent.invoke(state, config);
    const lastMessage = result.messages[result.messages.length - 1];

    return {
        messages: [
            new HumanMessage({content: lastMessage.content, name: "Reporter"}),
        ],
    };
};


const workflow = new StateGraph(AgentState)
    .addNode("data_fetcher", dataFetcherNode)
    .addNode("analyst", analystNode)
    .addNode("reporter", reporterNode)
    .addNode("supervisor", supervisorNode);

members.forEach((member) => {
    workflow.addEdge(member, "supervisor");
});

workflow.addConditionalEdges(
    "supervisor",
    (x: typeof AgentState.State) => x.next,
);

workflow.addEdge(START, "supervisor");

const graph = workflow.compile();

(async () => {
    const stockSymbol = 'EPAM';
    let streamResults = graph.stream(
        {
            messages: [
                new HumanMessage({
                    content: `Please provide an analysis of the stock symbol ${stockSymbol}.`,
                }),
            ],
        },
        {recursionLimit: 100},
    );

    for await (const output of await streamResults) {
        if (!output?.__end__) {
            console.log(output);
            console.log("----");
        }
    }
})();
