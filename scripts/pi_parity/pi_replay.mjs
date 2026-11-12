// Pi TS 一侧的回放：用 pi-agent-core 和 pi-ai 的 Chat Completions 连接器请求本地服务。
// 本地服务不校验密钥，这里的 apiKey 只是占位，Pi 没有密钥时不发请求。
import { Agent } from "@earendil-works/pi-agent-core";
import { createModels, createProvider } from "@earendil-works/pi-ai";
import { openAICompletionsApi } from "@earendil-works/pi-ai/api/openai-completions.lazy";
import { writeFileSync } from "node:fs";

const baseUrl = `http://127.0.0.1:${process.argv[2]}/pi/v1`;
const model = {
  id: "deepseek-v4.1-flash-1017", name: "replay", api: "openai-completions", provider: "replay",
  baseUrl, reasoning: false, input: ["text"],
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, contextWindow: 128000, maxTokens: 4096,
};
const provider = createProvider({
  id: "replay", name: "Replay", baseUrl,
  auth: { apiKey: { name: "Replay", resolve: async () => ({ auth: { apiKey: "local-replay" } }) } },
  models: [model], api: openAICompletionsApi(),
});
const models = createModels();
models.setProvider(provider);

const RULES = { "甬交规〔2026〕7号": "城区主干道限速 50 公里/小时", "甬交规〔2026〕9号": "城区次干道限速 40 公里/小时" };
const executed = [];
const lookup = {
  name: "lookup_rule", label: "查规定", description: "按文号查询交通规定的内容。",
  parameters: { type: "object", properties: { code: { type: "string" } }, required: ["code"] },
  async execute(id, params) {
    executed.push([id, params.code]);
    return { content: [{ type: "text", text: RULES[params.code] ?? "没有找到" }], details: {} };
  },
};
const agent = new Agent({
  initialState: { systemPrompt: "你是宁波市交通运输局的业务助手，用中文简洁作答。", model: models.getModel("replay", model.id), tools: [lookup] },
  streamFn: models.streamSimple.bind(models),
});
const events = [];
agent.subscribe((e) => {
  const m = e.message;
  if (e.type === "message_update") {
    const a = e.assistantMessageEvent;
    let extra = a.delta !== undefined ? JSON.stringify(a.delta) : "";
    if (a.type === "toolcall_start") { const tc = a.partial.content[a.contentIndex]; extra = `${tc.id} ${tc.name}`; }
    events.push(`message_update ${a.type}@${a.contentIndex} ${extra}`.trim());
  } else if (e.type === "message_start" || e.type === "message_end") {
    events.push(`${e.type} ${m.role}${m.role === "toolResult" ? " " + m.toolCallId : ""}`);
  } else if (e.type.startsWith("tool_execution")) {
    events.push(`${e.type} ${e.toolCallId}${e.type === "tool_execution_end" ? " error=" + e.isError : ""}`);
  } else {
    events.push(e.type);
  }
});
await agent.prompt("请用 lookup_rule 工具分别查询甬交规〔2026〕7号和甬交规〔2026〕9号的规定，然后回答两者的限速相差多少公里每小时。");
const messages = agent.state.messages.map((m) => ({
  role: m.role === "toolResult" ? "tool_result" : m.role,
  text: typeof m.content === "string" ? m.content : m.content.filter((b) => b.type === "text").map((b) => b.text).join(""),
  tool_calls: m.role === "assistant" ? m.content.filter((b) => b.type === "toolCall").map((b) => [b.id, b.name, b.arguments]) : undefined,
  tool_call_id: m.toolCallId, is_error: m.isError, stop_reason: m.stopReason, error: m.errorMessage,
  usage: m.usage ? { input: m.usage.input, output: m.usage.output, cache_read: m.usage.cacheRead, total: m.usage.totalTokens } : undefined,
}));
writeFileSync(`${process.argv[3]}/pi-result.json`, JSON.stringify({ events, executed, messages }, null, 1));
console.log("Pi TS 完成：事件", events.length, "个，工具执行", executed.length, "次");
