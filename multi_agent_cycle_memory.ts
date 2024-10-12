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

(async () => {
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
        description: "Fetches the latest stock price and previous close price for a given stock symbol.",
        schema: z.object({
            symbol: z.string().describe("Stock symbol to fetch price for"),
        }),
        func: async ({symbol}) => {
            //  data: {
            //     'Global Quote': {
            //       '01. symbol': 'EPAM',
            //       '02. open': '191.9900',
            //       '03. high': '194.1400',
            //       '04. low': '191.5172',
            //       '05. price': '192.7700',
            //       '06. volume': '343878',
            //       '07. latest trading day': '2024-10-10',
            //       '08. previous close': '194.1500',
            //       '09. change': '-1.3800',
            //       '10. change percent': '-0.7108%'
            //     }
            //   }
            const alphaVantageApiKey = process.env.ALPHA_VANTAGE_API_KEY;
            const url = `https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=${symbol}&apikey=${alphaVantageApiKey}`;
            const response = await axios.get(url);
            const stockPrice = response.data['Global Quote']['05. price'];
            const previousPrice = response.data['Global Quote']['08. previous close'];
            if (!stockPrice) {
                throw new Error(`Could not fetch stock price for symbol: ${symbol}`);
            }

            return {stockPrice, previousPrice, symbol};
        },
    });

    const members = ["data_fetcher", "analyst", "investor", "reporter", "waiter"] as const;

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
        .pipe(
            llm.bindTools([routingTool], {
                tool_choice: "route",
            })
        )
        .pipe(new JsonOutputToolsParser())
        .pipe((x) => x[0].args);

    const supervisorNode = async (
        state: typeof AgentState.State,
        config?: RunnableConfig
    ) => {
        const result = await supervisorChain.invoke(state, config);
        let next = result.next;

        if (state.memory.outlook === 'neutral' && next === 'investor') {
            console.log('Neutral outlook received. Redirecting to wait before re-evaluation...');
            next = 'waiter';
        }

        return {
            next: next,
        };
    };

    const dataFetcherAgent = createReactAgent({
        llm,
        tools: [stockPriceTool],
        messageModifier: new SystemMessage(
            "You are a data fetcher. Use the get_stock_price tool to fetch the latest and previous stock prices for the requested stock symbol. Provide the data in the exact format: 'symbol: SYMBOL, latestPrice: PRICE, previousPrice: PRICE'. Do not include any additional text or analysis."
        ),
    });

    const dataFetcherNode = async (
        state: typeof AgentState.State,
        config?: RunnableConfig
    ) => {
        const result = await dataFetcherAgent.invoke(state, config);
        const lastMessage = result.messages[result.messages.length - 1];

        const stockSymbolMatch = lastMessage.content.match(
            /symbol:\s*(\w+),\s*latestPrice:\s*([\d\.]+),\s*previousPrice:\s*([\d\.]+)/
        );
        let stockSymbol = "";
        let latestPrice = 0;
        let previousPrice = 0;

        if (stockSymbolMatch) {
            stockSymbol = stockSymbolMatch[1];
            latestPrice = parseFloat(stockSymbolMatch[2]);
            previousPrice = parseFloat(stockSymbolMatch[3]);
        }

        return {
            messages: [
                new HumanMessage({
                    content: lastMessage.content,
                    name: "DataFetcher",
                }),
            ],
            memory: {
                stockSymbol,
                latestPrice,
                previousPrice,
            },
        };
    };

    const analystAgent = createReactAgent({
        llm,
        tools: [],
        messageModifier: new SystemMessage(
            `You are a financial analyst.

Use the latest and previous stock prices provided in the conversation to analyze the stock's trend.

Determine the outlook (positive, negative, neutral) based on the price movement and make a recommendation.

If the outlook is 'neutral', inform that a re-evaluation is needed after some time.`
        ),
    });

    const analystNode = async (
        state: typeof AgentState.State,
        config?: RunnableConfig
    ) => {
        const {latestPrice, previousPrice, stockSymbol} = state.memory;

        const analysisPrompt = `Analyze the stock symbol ${stockSymbol}.

Latest Price: $${latestPrice}
Previous Price: $${previousPrice}

Provide the outlook (positive, negative, neutral) and a recommendation.`;

        const result = await analystAgent.invoke({
            messages: [
                ...state.messages,
                new HumanMessage({content: analysisPrompt, name: "Analyst"}),
            ],
        }, config);
        const lastMessage = result.messages[result.messages.length - 1];

        const outlookMatch = lastMessage.content.match(/Outlook:\s*(positive|negative|neutral)/i);
        let outlook = "neutral";
        if (outlookMatch) {
            outlook = outlookMatch[1].toLowerCase();
        }

        return {
            messages: [
                new HumanMessage({content: lastMessage.content, name: "Analyst"}),
            ],
            memory: {
                analysis: lastMessage.content,
                outlook,
            },
        };
    };

    const investorAgent = createReactAgent({
        llm,
        tools: [],
        messageModifier: new SystemMessage(
            "You are an investor. Based on the analyst's recommendation, decide to BUY, SELL, or HOLD the stock. Simulate the action with console.log."
        ),
    });

    const investorNode = async (
        state: typeof AgentState.State,
        config?: RunnableConfig
    ) => {
        const {analysis, stockSymbol} = state.memory;

        const investorPrompt = `Based on the following analysis, decide to BUY, SELL, or HOLD the stock ${stockSymbol}.

${analysis}

Provide your decision and simulate the action with console.log.`;

        const result = await investorAgent.invoke({
            messages: [
                ...state.messages,
                new HumanMessage({content: investorPrompt, name: "Investor"}),
            ],
        }, config);
        const lastMessage = result.messages[result.messages.length - 1];

        const decisionMatch = lastMessage.content.match(/Decision:\s*(BUY|SELL|HOLD)/i);
        let decision = "HOLD";
        if (decisionMatch) {
            decision = decisionMatch[1].toUpperCase();
        }

        console.log(`Investor Decision: ${decision} ${stockSymbol}`);
        if (decision === "BUY") {
            console.log(`Simulating BUY action for ${stockSymbol}`);
        } else if (decision === "SELL") {
            console.log(`Simulating SELL action for ${stockSymbol}`);
        } else {
            console.log(`Holding position for ${stockSymbol}`);
        }

        return {
            messages: [
                new HumanMessage({content: lastMessage.content, name: "Investor"}),
            ],
        };
    };

    const reporterAgent = createReactAgent({
        llm,
        tools: [],
        messageModifier: new SystemMessage(
            "You are the reporter. Summarize the analysis and investor's decision and provide it to the user."
        ),
    });

    const reporterNode = async (
        state: typeof AgentState.State,
        config?: RunnableConfig
    ) => {
        const result = await reporterAgent.invoke(state, config);
        const lastMessage = result.messages[result.messages.length - 1];

        return {
            messages: [
                new HumanMessage({content: lastMessage.content, name: "Reporter"}),
            ],
        };
    };

    const waiterNode = async (
        state: typeof AgentState.State,
        config?: RunnableConfig
    ) => {
        console.log('Neutral outlook received. Waiting for 1 minute before re-evaluation...');
        let seconds = 60;
        const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));
        while (seconds > 0) {
            process.stdout.write(`\rTime remaining: ${seconds--} seconds`);
            await sleep(1000);
        }
        process.stdout.write('\n');
        return {};
    };

    const workflow = new StateGraph(AgentState)
        .addNode("data_fetcher", dataFetcherNode)
        .addNode("analyst", analystNode)
        .addNode("investor", investorNode)
        .addNode("reporter", reporterNode)
        .addNode("supervisor", supervisorNode)
        .addNode("waiter", waiterNode);

    members.forEach((member) => {
        workflow.addEdge(member, "supervisor");
    });

    workflow.addEdge("waiter", "data_fetcher");
    workflow.addConditionalEdges("supervisor", (state: typeof AgentState.State) => state.next);
    workflow.addEdge(START, "supervisor");

    const graph = workflow.compile();

    const stockSymbol = 'EPAM';
    let streamResults = graph.stream(
        {
            messages: [
                new HumanMessage({
                    content: `Please provide an analysis of the stock symbol ${stockSymbol}.`,
                }),
            ],
        },
        {recursionLimit: 200}
    );

    for await (const output of await streamResults) {
        if (!output?.__end__) {
            console.log(output);
            console.log("----");
        }
    }
})();
