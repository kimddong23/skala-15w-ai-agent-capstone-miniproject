/** 재사용된 백엔드의 LLM 모드가 기대와 다르면 즉시 실패시킨다 (결정론 스위트가 LLM 서버에 붙는 사고 방지). */
const BACKEND = Number(process.env.E2E_BACKEND_PORT ?? 8100);

export default async function globalSetup() {
  const want = process.env.LLM_ENABLED === "true";
  const h = await (await fetch(`http://127.0.0.1:${BACKEND}/health`)).json();
  if (!h.mcp) throw new Error(`백엔드 MCP 미연결: ${JSON.stringify(h)}`);
  if (h.llm !== want) throw new Error(`백엔드 llm=${h.llm} 이지만 이번 실행은 LLM_ENABLED=${want}. ${BACKEND} 포트의 기존 서버를 확인하세요.`);
}
