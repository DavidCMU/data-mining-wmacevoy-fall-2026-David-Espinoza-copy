export const meta = {
  name: 'zen-coder-eval',
  description: 'Run Zen Coder conversations across system prompts and retrieval depth k, then judge each one',
  whenToUse: 'Comparing Zen Coder system prompts and how many guide sections (k) to retrieve per turn',
  phases: [
    { title: 'Converse', detail: 'Zen Coder (sonnet) talks with a simulated novice (haiku)', model: 'sonnet' },
    { title: 'Judge', detail: 'opus scores each conversation against the rubric', model: 'opus' },
  ],
}

// Zen Coder test harness. Each conversation is one (prompt, k, scenario, repeat):
//
//   turn 1   retrieve k sections for the novice's opening, then reply
//   turn n   retrieve k sections for the query Zen Coder asked for last turn, then reply
//
// Zen Coder replies with {response, query, context}; the query feeds the next
// turn's retrieval. k = 0 is the control: no retrieval at all.
//
// This is the no-API-key way to run the harness, through Claude Code's own models.
// The maintained path is zen/run.py with the Claude API; prompts.json and scenarios.json
// are the source of truth there. A workflow script can't read files, so PROMPTS and
// SCENARIOS below are a copy (as of run 2026-10-22c): when you change one, change both.
//
// args (all optional): {run, prompts, ks, scenarios, repeats, turns, repo}
// Records land in <repo>/llm/data/zen/<run>/; summarize with `pixi run -e llm zen-score <run>`.

const A = args || {}
const REPO = A.repo || '/Users/wmacevoy/projects/data-mining-wmacevoy-fall-2026'
const RUN = A.run || 'run'
const OUT = `${REPO}/llm/data/zen/${RUN}`
const TURNS = A.turns || 3

const LEAD = 'You are Zen Coder, a guide to a novice developer. You want the developer to learn how to build good systems.'

const PROMPTS = {
  // The prompt the toy chatbot started with.
  baseline: `${LEAD}
Find out what the user wants to build, then recommend good ways to build it and explain why. Be calm, clear, and concise. Ask clarifying questions when the goal is vague, but no more than one or two at a time. Favor simple approaches a beginner can understand and grow from, and explain the reasoning so they learn the principle, not just the answer.`,

  // Ask first, advise later.
  socratic: `${LEAD}
Do not advise until you know three things: where it will run, what the developer already knows, and what "working" would look like to them. Ask for those one or two at a time. Once you know them, give the single simplest next step and the reason for it. Let the developer do the thinking; your questions are the teaching.`,

  // Every recommendation carries its principle.
  principle: `${LEAD}
Find out what the user wants to build, asking at most two questions at a time. Every recommendation you make has three parts: the step, a question or small experiment that lets the developer see for themselves why it matters, and something they can do today to start. Prefer the simplest version that works end to end, and grow it from there. Be calm and concise.`,
}

const SCENARIOS = {
  alarm: {
    opening: 'I want to make a alarm clock for my classes that is not distracting.',
    persona: 'A first-year college student who took one intro Python course.',
    facts: [
      'Wants it to run on a MacBook, not a phone.',
      'Has written small Python scripts but never an app.',
      '"Not distracting" means one quiet notification five minutes before class, nothing else.',
      'Classes differ by day: Monday/Wednesday/Friday at 9:00, Tuesday/Thursday at 11:00.',
      'Wants it working by next week.',
    ],
    relevant: ['guide/08-work.md#w6', 'guide/05-design.md#d5', 'guide/06-testing.md#t7',
               'guide/04-requirements.md#r5', 'guide/04-requirements.md#r1'],
  },
  club: {
    opening: 'My club needs a website where members can sign up for events.',
    persona: 'A sophomore who is club treasurer and has built a static HTML page once.',
    facts: [
      'About 40 members; events have a capacity limit.',
      'The club has no budget.',
      'Sign-ups would collect names and student emails.',
      'Club officers change every year, and someone else will have to run it.',
      'Has never deployed anything to the internet.',
    ],
    relevant: ['guide/07-operating.md#o8', 'guide/04-requirements.md#r8',
               'guide/07-operating.md#if-you-have-never-deployed-anything',
               'guide/05-design.md#d12', 'guide/02-unknowing.md#u6', 'guide/08-work.md#w6'],
  },
  grades: {
    opening: 'I had an AI write me an app that tracks my grades but now it is broken and I do not know why.',
    persona: 'A non-CS major who has been using an AI assistant to write all of the code.',
    facts: [
      'It is a Python script of about 300 lines that they did not write themselves.',
      'It broke after they asked the AI to "add weighted categories."',
      'There are no tests; they check it by running it and looking at the numbers.',
      'They do not use git.',
    ],
    relevant: ['guide/10-agents.md#a2', 'guide/10-agents.md#a1', 'guide/06-testing.md#t3',
               'guide/10-agents.md#a7', 'guide/06-testing.md#t1', 'guide/06-testing.md#t9'],
  },
  plant: {
    opening: 'I want my Raspberry Pi to text me when my plant needs water.',
    persona: 'A hobbyist who has followed a few Raspberry Pi tutorials.',
    facts: [
      'Already has a soil-moisture sensor wired up and can print readings.',
      'The readings sometimes jump to nonsense values for a moment.',
      'The Wi-Fi in the apartment drops out at night.',
      'Would be upset to get ten texts in a row, or none when the plant is dying.',
    ],
    relevant: ['guide/04-requirements.md#r6', 'guide/07-operating.md#o4', 'guide/07-operating.md#o5',
               'guide/03-feedback.md#f4', 'guide/08-work.md#w6'],
  },
}

const PROMPT_KEYS = A.prompts || ['baseline', 'principle']
const KS = A.ks || [0, 3, 5]
const SCENARIO_KEYS = A.scenarios || ['alarm', 'plant']
const REPEATS = A.repeats || 1

// Principle IDs that exist in the guide (SE1 v1.1): letter -> count.
const COUNTS = { P: 5, U: 7, F: 5, R: 11, D: 13, T: 11, O: 11, W: 12, H: 12, A: 10 }
const VALID = new Set(Object.entries(COUNTS).flatMap(([p, n]) =>
  Array.from({ length: n }, (_, i) => `SE1/${p}${i + 1}`)))
const CITED = /\b(?:SE1\/)?([PUFRDTOWHA]\d{1,2})\b/g
const SHOWN = /\bthe guide\b|\bretriev\w*/gi   // naming its sources to the developer

const ZEN_SCHEMA = {
  type: 'object',
  properties: {
    response: { type: 'string', description: 'exactly what the novice will see' },
    query: { type: 'string', description: 'a short search query into the guide for your next turn' },
    context: {
      type: 'array',
      description: 'each section the retrieval command printed, in order; empty if none',
      items: { type: 'object', properties: { ref: { type: 'string' }, heading: { type: 'string' } }, required: ['ref', 'heading'] },
    },
  },
  required: ['response', 'query', 'context'],
}

const SCORE = { type: 'integer', minimum: 1, maximum: 5 }
const JUDGE_SCHEMA = {
  type: 'object',
  properties: {
    understands: SCORE, fits: SCORE, teaches: SCORE, grounded: SCORE, tone: SCORE,
    facts_discovered: { type: 'array', items: { type: 'string' }, description: 'hidden facts the novice revealed' },
    first_recommendation_turn: { type: 'integer', minimum: 0, description: 'first Zen turn with a concrete next step; 0 if none' },
    verdict: { type: 'string', description: 'one or two sentences' },
  },
  required: ['understands', 'fits', 'teaches', 'grounded', 'tone', 'facts_discovered', 'first_recommendation_turn', 'verdict'],
}

function transcript(turns, { withContext = false } = {}) {
  return turns.map((t, i) => {
    let s = `Novice: ${t.user}\n`
    if (withContext && t.context.length)
      s += `[retrieved for turn ${i + 1}: ${t.context.map(c => `${c.ref} "${c.heading}"`).join('; ')}]\n`
    else if (withContext) s += `[nothing retrieved for turn ${i + 1}]\n`
    return s + (t.response ? `Zen Coder: ${t.response}\n` : '')
  }).join('\n')
}

// Zen Coder knows the guide but never shows it: no IDs, no quotes, no "the guide
// says". It leads the novice through the door instead of pointing at the sign.
const KNOWING = `What you know: a software-engineering guide, condensed. Read it before every reply with
cat ${REPO}/llm/zen/guide-summary.md
It is your private understanding, not a reference for the developer. Never cite it: no principle IDs (like SE1/T3), no quoting it, no "the guide says". Lead the developer through the door rather than pointing at the sign: ask the question, or set the small step, that lets them discover the principle for themselves, in plain words.`

function zenPrompt(c, history, userMsg, query) {
  const retrieval = c.k === 0
    ? 'Use no other tools.'
    : `Then, to go deeper on this turn, run this retrieval command once, exactly as written:

cd ${REPO} && pixi run -e llm python llm/zen/retrieve.py -k ${c.k} 2>/dev/null <<'ZEN_QUERY'
${query}
ZEN_QUERY

It prints the ${c.k} sections of the full guide most relevant to the query. They are private background like the summary: use what helps this developer, ignore what doesn't, and never mention, quote, or cite them. Use no other tools.`

  return `${PROMPTS[c.prompt]}

${KNOWING}

${retrieval}

Conversation so far:
${transcript(history) || '(this is the first message)'}
Novice: ${userMsg}

Return:
- response: your reply, exactly as the novice will see it.
- query: a short search query into the guide that would help you on your next turn.
- context: the ref and heading of each section the retrieval command printed (empty if you ran none).`
}

function userPrompt(sc, history) {
  return `This is a role-play inside an evaluation. You are NOT a coding assistant here: do not look at, mention, or report on any files, repositories, tools, or test harnesses. Your only output is a novice developer's next chat message to a mentor called Zen Coder. Stay in character, and use no tools.

Who you are: ${sc.persona}
Facts about your situation. Reveal one only when Zen Coder asks about it or it naturally comes up; never volunteer them all at once:
${sc.facts.map(f => `- ${f}`).join('\n')}

Conversation so far:
${transcript(history)}
Write your next message to Zen Coder: 1–3 casual sentences, the way a novice would write. Never mention that this is a simulation. Reply with the message only.`
}

function metrics(sc, turns) {
  const retrieved = new Set()
  let leaks = 0, questions = 0, words = 0
  for (const t of turns) {
    for (const c of t.context) retrieved.add(c.ref)
    // A bare valid ID counts; a bare invalid one is likely a pin name ("A0").
    leaks += [...t.response.matchAll(CITED)].filter(m => m[0].startsWith('SE1/') || VALID.has(`SE1/${m[1]}`)).length
    leaks += (t.response.match(SHOWN) || []).length
    questions += (t.response.match(/\?/g) || []).length
    words += t.response.split(/\s+/).filter(Boolean).length
  }
  const found = sc.relevant.filter(r => retrieved.has(r)).length
  return {
    relevant_recall: found / sc.relevant.length,
    guide_leaks: leaks,
    questions_per_turn: questions / turns.length, words_per_turn: words / turns.length,
  }
}

// The simulated novice runs as an ordinary subagent that can see this repository,
// and it sometimes breaks character to report on it ("the harness already exists,
// untracked, in .../llm/zen/"). One such turn contaminates the whole conversation,
// so retry once and then give up rather than score it.
const LEAK = /\/Users\/|\bharness\b|\buntracked\b/i

async function novice(c, sc, turns, n) {
  for (let attempt = 1; attempt <= 2; attempt++) {
    const u = await agent(userPrompt(sc, turns),
      { label: `${c.id} novice ${n}${attempt > 1 ? ' retry' : ''}`, phase: 'Converse', model: 'haiku', effort: 'low' })
    const msg = u && String(u).trim()
    if (msg && !LEAK.test(msg)) return msg
    log(`${c.id}: novice turn ${n} broke character (attempt ${attempt})`)
  }
  throw new Error(`${c.id}: novice turn ${n} broke character twice`)
}

async function converse(c) {
  const sc = SCENARIOS[c.scenario]
  const turns = []
  let userMsg = sc.opening, query = sc.opening
  for (let n = 1; n <= TURNS; n++) {
    const z = await agent(zenPrompt(c, turns, userMsg, query),
      { label: `${c.id} zen ${n}`, phase: 'Converse', model: 'sonnet', schema: ZEN_SCHEMA })
    if (!z) throw new Error(`${c.id}: Zen Coder turn ${n} failed`)
    turns.push({ user: userMsg, query: c.k ? query : null, context: c.k ? z.context : [],
                 response: z.response, next_query: z.query })
    if (n === TURNS) break
    userMsg = await novice(c, sc, turns, n)
    query = z.query
  }
  return { ...c, run: RUN, turns, metrics: metrics(sc, turns) }
}

async function judge(rec) {
  const sc = SCENARIOS[rec.scenario]
  const file = `${OUT}/${rec.id}.json`
  const scores = await agent(`You are judging a conversation between Zen Coder, a mentor bot for novice developers, and a simulated novice. Its goal: learn what the novice wants to build, then recommend good ways to build it and explain why, so the novice learns to build good systems.

The novice: ${sc.persona}
Hidden facts the novice would reveal only if asked:
${sc.facts.map(f => `- ${f}`).join('\n')}
Guide sections a strong answer would likely draw on: ${sc.relevant.join(', ')}

Transcript (bracketed lines show what Zen Coder retrieved from the guide before each reply; the novice did not see them):
${transcript(rec.turns, { withContext: true })}

Score each criterion 1 (poor) to 5 (excellent). Think through the evidence before each score.
- understands: asks the questions that matter for this project, no more than one or two at a time, and doesn't interrogate.
- fits: recommends the simplest workable path for this novice and their actual constraints.
- teaches: leads the novice through the door rather than pointing at the sign. A question they can answer, or a small step they take, lets them discover why it matters, so they could reuse the idea on the next project. Lecturing a principle at them scores lower.
- grounded: the advice is consistent with the guide (Zen Coder knows it, and you can read it: cat ${REPO}/llm/zen/guide-summary.md), anything drawn from retrieved sections is used correctly, and irrelevant sections are ignored rather than forced in. Zen Coder must never show its sources: any principle ID, quotation, or mention of a guide or retrieval in a reply is a serious fault here.
- tone: calm, concise, encouraging, not preachy.
Also list the hidden facts the novice actually revealed, and the first Zen Coder turn (1-based) with a concrete next step.

Before returning, save the conversation record: run \`mkdir -p ${OUT}\`, then use the Write tool to write this JSON to ${file}, exactly as given:
${JSON.stringify(rec)}
Then write your scores, as JSON with the same fields you return, to ${OUT}/${rec.id}.judge.json.`,
    { label: `${rec.id} judge`, phase: 'Judge', model: 'opus', schema: JUDGE_SCHEMA })
  return scores && { id: rec.id, prompt: rec.prompt, k: rec.k, scenario: rec.scenario, ...rec.metrics, ...scores }
}

const convs = []
for (const prompt of PROMPT_KEYS)
  for (const k of KS)
    for (const scenario of SCENARIO_KEYS)
      for (let r = 1; r <= REPEATS; r++)
        convs.push({ id: `${prompt}-k${k}-${scenario}-r${r}`, prompt, k, scenario, repeat: r })

log(`${convs.length} conversations (${PROMPT_KEYS.length} prompts × k ∈ {${KS}} × ${SCENARIO_KEYS.length} scenarios × ${REPEATS}), ${TURNS} Zen turns each → ${OUT}`)

const rows = await pipeline(convs, c => converse(c), rec => judge(rec))
const done = rows.filter(Boolean)
if (done.length < convs.length) log(`${convs.length - done.length} conversation(s) failed and are missing from the results`)
return { run: RUN, out: OUT, rows: done }
