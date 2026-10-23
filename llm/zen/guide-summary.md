# Software Engineering (SE1), condensed

A condensed form of the guide at github.com/wmacevoy/software-engineering (v1.1, commit de81f7c),
for use as leading context in a prompt. Principles are cited as SE1/<ID> (for example SE1/T3).
Each line is the principle's own headline, then its gist, usually the question the guide says to ask.

## The argument

1. Engineering is making reliable things on time, within constraints, while managing everyone's
   discomfort. Reliable means it works, and says so when it doesn't.
2. Every practice retires a named fear. Relieving the fear without reducing the risk is worse than nothing.
3. Building is discovering what you didn't know, in the right order, at the right price. The
   dangerous kind is confident and wrong.
4. Software's tests are cheap and software has no mass: certainty is affordable, and necessary,
   because a bad design never visibly sags.
5. Nobody hands you the problem. Write requirements specific enough to be wrong; they always include
   security, law, and ethics.
6. Every input can be absent, late, wrong, or lying, and each case needs a stated answer.
7. State is the material. Name your caches, own your copies, keep logic away from I/O.
8. A test you have never seen fail proves nothing; a flaky one is worse than none.
9. Choose how carefully to work per decision by the cost of being wrong once.
10. Nothing important may live only in a head: teammates leave, and agents were never in the room.

## Kinds of not-knowing

- K0, the known: true and known. Write it down; tests are executable knowledge.
- K1, the known unknown: you know the question. Cheap: look it up, spike, ask.
- K2, the unknown unknown: arrives as a surprise. Only exposure to reality (integrate, deploy, demo, ask a domain expert) finds it, so arrange to be surprised early.
- K3, erroneous certainty: feels like K0 and is false. The most expensive; only evidence (a test that could fail, a second person) finds it.
- K4, chosen falsehood: held as true on purpose because it is useful. Write down where it is load-bearing.

## 1. Purpose

- P1. Engineers make reliable things, on time, within resource constraints, managing discomfort for the team, the client, and the users. Which of these is fixed, and which moves when we are wrong?
- P2. Reliable is not correct. When this is broken, how does anyone find out?
- P3. Discomfort is a leading indicator. Who is uncomfortable right now, and what are they afraid of?
- P4. Every practice retires a named fear. Does it retire the risk, or only the fear?
- P5. The deliverable is not the code. Could a stranger run, reset, and fix this from what is in the repository?

## 2. Unknowing

- U1. K0 through K3 are in order of expense; K4 is a choice. A K2 costs a rewrite; a K3 costs everything built on the belief.
- U2. Building is discovering, in the right order and at the right price, which category each part of the problem belongs to. What is the cheapest way to move this part?
- U3. K3 is the one to be afraid of. What would I see if I were wrong about this, and have I looked?
- U4. Every abstraction rests on chosen falsehoods; write down where they are load-bearing.
- U5. Estimate the category, not the task. K0 gets a number, K1 a range, K2 only a time box.
- U6. Nothing important may live only in a head. If the person who knows this vanished on Friday, what couldn't we do on Monday?
- U7. Find the K3 factories: anything that keeps saying "fine" with no way to say "wrong".

## 3. Feedback

- F1. The cost of a test decides the method. What does it cost to find out we are wrong, and to undo it?
- F2. The cost of a test is designed, not given. What would make this cheap to test?
- F3. Push each class of error down to the cheapest loop that can catch it.
- F4. Nothing has mass, so build the signals. If this design were wrong, what would tell us, and when?
- F5. When a measure becomes a target, it stops measuring. What is the cheapest way to fake it?

## 4. Requirements

- R1. Keep the pain; test the solution. What happens today when this goes wrong, and who gets the call?
- R2. Specific and wrong beats vague and safe. Could the client read this sentence and say "no"?
- R3. Ask the questions nobody volunteers, the ones that could invalidate the plan, first. None are about features.
- R4. A requirement with no price attached is a wish. What does this sentence cost, and who agreed to pay it?
- R5. A requirement says what must be observably true; a design says how. Could another team satisfy this a different way?
- R6. Every input has four failure cases: absent, late, wrong, and lying. Each needs a stated answer.
- R7. Scale, recovery, and availability are answered whether or not anyone writes them down. How many in two years, how fast at the peak, what happens when it breaks?
- R8. Security, law, and ethics are always requirements. Who, outside this room, does this system keep a record about?
- R9. Write down what you decided not to do, and where someone would find it.
- R10. Version requirements; never edit them in place. The diff between v1 and v2 is the most useful conversation with the client.
- R11. An API is the set of observables your acceptance tests need.

## 5. Design

- D1. State is the material. How much state can this change reach?
- D2. Anything that might be out of date is a cache. What is this a copy of, and how stale can it get?
- D3. There is no single "now". How wrong may this copy be, and who finds out when it is?
- D4. Make operations idempotent. What happens if this request is sent twice?
- D5. Wall-clock time and elapsed time are different instruments. Which clock is this, and what happens when it jumps?
- D6. Derive instead of store; name the authority; record why.
- D7. Serve, don't rule: keep the logic away from the I/O. The rule is a testable function; I/O serves it.
- D8. Separate essential complexity from accidental complexity. What requirement would have to disappear for this to go away?
- D9. Conceptual integrity: one coherent design beats every good idea.
- D10. Locality of reasoning: use a part correctly without reading inside it, and its contract says how it fails.
- D11. Once someone depends on an interface, it is a promise. Is this a one-way door?
- D12. Software is capital you carry. What does this cost us every week for two years, and what can we delete?
- D13. Provenance: who said this, and can I check it later without asking them again?

## 6. Testing

- T1. A test is a fear with a name. Which fear does it retire, and have I seen it fail?
- T2. Climb the reliability ladder: a reliable thing checks itself and says so when it is broken.
- T3. A test that cannot fail proves nothing. Have I seen this test fail, and this instrument fire?
- T4. A loop you do not trust is worse than none. When it goes red, does everyone believe it?
- T5. Repeatability is a property of the state a test touches. What does this test write, and who else reads it?
- T6. The mock and the real thing answer to the same test.
- T7. Test user interfaces by pulling the logic out first.
- T8. When you don't know how to build it, write the tests that would pass when it works.
- T9. A bug you cannot reproduce is a rumor. Can I make it happen on purpose?
- T10. Let the machine generate the cases you would not think of. What rule must hold for every input?
- T11. When someone else owns "correct," test against their implementation.

## 7. Operating

- O1. Repeatability is also a property of the environment.
- O2. In the manifest, say what you are compatible with; in the lock, record what you ran; commit both.
- O3. CI runs the command you run, everywhere you ship, and enforces what matters.
- O4. Every failure needs a signal a person will see, and believe. If this broke at 3am, what would tell us?
- O5. Handle failure on purpose: timeouts, bounded retries, and a stated fallback.
- O6. A deploy is a test whose failure is an outage. Has the rollback been run, and how long did it take?
- O7. Change data and interfaces in steps that can each be undone.
- O8. Secure by default. What can an attacker reach from here, and have we tried?
- O9. In an incident, restore service first; learn afterward.
- O10. Measure before you optimize; test at the scale you promised.
- O11. A release you cannot roll back is a one-way door; ship it like one.

## 8. Work

- W1. Every method runs the same six steps; they differ in batch size and in where the tests are.
- W2. Waterfall is correct when the only real test is expensive.
- W3. Iterative work puts a test inside every step.
- W4. Choose how carefully to work per decision, by the cost of being wrong once. Loop fast around reversible choices; slow down at one-way doors.
- W5. Flatten the schedule: a dependency is an unwritten interface.
- W6. Build a thin vertical slice first, then thicken it. One path end to end beats finished horizontal layers.
- W7. Deadlines produce an avoidance gradient; change the terrain, not the willpower. What would make the last week boring?
- W8. Don't fall down the waterfall. What will I know when this spike stops?
- W9. An iteration is time-boxed, ends on a fence, and produces four things, starting with a running thing the client can touch. When time runs out, scope moves, not the date.
- W10. Keep the working set in the repository, versioned with the code: not in a wiki, chat, or notebook.
- W11. Done means all of it, one concern per change, reviewed by someone else. Could I review this in one sitting?
- W12. Joining late: rebuild the working set, and mark what you inferred.

## 9. People

- H1. The team is a first-order variable.
- H2. A teammate fails the way an input fails: absent, late, wrong, or lying. Plan for it.
- H3. Roles exist to retire fears; rotate them so knowledge spreads.
- H4. The interface between two people is a contract.
- H5. The client knows the domain, and cannot see its edges.
- H6. Read your assumptions to the client: "We're assuming X. True?"
- H7. The first conversation: pain before solution, people before features, numbers before nouns.
- H8. A demo is a test, so run it like one: the real system, the client drives, write down every "huh."
- H9. With bad news, the delay is the cost.
- H10. Settle disagreements by category, then by reversibility.
- H11. Being behind is a systems failure with a schedule. Don't add work to the person with the most knowledge.
- H12. Fix the process, not the person. Nobody is blamed for a K2.

## 10. Agents

- A1. Producing code got cheaper faster than knowing it is right.
- A2. An agent is a K3 factory; never let confidence substitute for a run. Was this run, or only reasoned about?
- A3. An agent arrives knowing nothing and leaves remembering nothing. What would a competent stranger need to read?
- A4. Context is a budget, and it resets every session.
- A5. Spend the speed on lowering the cost of testing: fixtures, harnesses, seed data.
- A6. Review capacity is the bottleneck. Ask for the smallest change that does the job.
- A7. Delegation is not abdication. Working code nobody can explain is a liability.
- A8. Agents move the human up the waterfall, toward people, laws, and judgment.
- A9. Hand it the search-shaped work; keep the decisions.
- A10. Ask your agents, and write the answers down.
