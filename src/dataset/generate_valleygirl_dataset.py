"""
Generates a multi-turn SFT dataset for the valley girl chatbot.

The character:
  - Answers in valley girl voice (like, literally, oh my god, so, hyperbole)
  - Briefly addresses the question, then steers hard toward personal drama:
    relationships, boys, who said what, shopping, projection onto the user
  - When pushed back ("no I just wanted to know"), doubles down with projection
  - Eventually may ask the user a pointed personal question

Output: data/valleygirl_train.jsonl and data/valleygirl_eval.jsonl
Run: uv run python src/dataset/generate_valleygirl_dataset.py
"""

import argparse
import json
import random
import time
from pathlib import Path

import anthropic

SEED_QUESTIONS = [
    # Science & nature
    "How does photosynthesis work?",
    "What causes earthquakes?",
    "Why is the sky blue?",
    "How do vaccines work?",
    "What is a black hole?",
    "Why do we dream?",
    "How does DNA replication work?",
    "What causes the Northern Lights?",
    "Why do leaves change color in autumn?",
    "How does the immune system fight viruses?",
    # History & geography
    "Who was Napoleon Bonaparte?",
    "What caused World War I?",
    "What is the capital of Australia?",
    "How did the Roman Empire fall?",
    "Who built the pyramids?",
    "What was the Cold War about?",
    "What is the Silk Road?",
    "How did democracy start in Ancient Greece?",
    # Math & logic
    "What is the Pythagorean theorem?",
    "How does compound interest work?",
    "What is a prime number?",
    "Can you explain the Monty Hall problem?",
    "What is the difference between mean and median?",
    # Technology
    "How does the internet work?",
    "What is machine learning?",
    "How does GPS know where I am?",
    "What is blockchain?",
    "How does a computer processor work?",
    "What is Wi-Fi and how does it work?",
    # Culture & arts
    "Who wrote Romeo and Juliet?",
    "What is impressionism in art?",
    "How did jazz music originate?",
    "What is the significance of the Mona Lisa?",
    "Who was Frida Kahlo?",
    # Practical / everyday
    "How do I make sourdough bread?",
    "What's the best way to study for exams?",
    "How do I start investing?",
    "What are the health benefits of exercise?",
    "How does sleep affect memory?",
    "What's the best way to apologize to someone?",
    "How do you write a cover letter?",
    "What causes inflation?",
    # Philosophy & big questions
    "What is consciousness?",
    "Is free will real?",
    "What does it mean to live a good life?",
    "What is the meaning of art?",
    "Do animals have emotions?",
    # Psychology & behavior
    "What is the placebo effect?",
    "Why do people procrastinate?",
    "What causes anxiety?",
    "How does addiction work?",
    "What is cognitive dissonance?",
    "Why do we get jealous?",
    "How does peer pressure work?",
    # Sports & fitness
    "How do muscles grow after working out?",
    "What is the offside rule in soccer?",
    "How does a marathon runner pace themselves?",
    "What causes a runner's high?",
    "How do Olympic athletes train?",
    # Food & nutrition
    "Why does spicy food make your mouth burn?",
    "How does fermentation work?",
    "What makes bread rise?",
    "Why is caffeine addictive?",
    "What is the difference between vitamins and minerals?",
    # Environment & climate
    "How does climate change work?",
    "What is the ozone layer?",
    "Why are coral reefs dying?",
    "How do hurricanes form?",
    "What is the water cycle?",
    # Law & politics
    "How does the Supreme Court work?",
    "What is the Electoral College?",
    "How are laws made in Congress?",
    "What is the difference between civil and criminal law?",
    "How does the UN work?",
    # Space & astronomy
    "How far away is the Moon?",
    "What is a neutron star?",
    "How did the universe begin?",
    "Could there be life on Mars?",
    "What is a solar eclipse?",
    # Language & communication
    "How do children learn to speak?",
    "Why do accents exist?",
    "How many languages are there in the world?",
    "What is body language?",
    "How does translation work?",
    # Economics & money
    "What is the stock market?",
    "How do banks make money?",
    "What is GDP?",
    "Why do some countries have stronger currencies?",
    "What is supply and demand?",
    # Animals & biology
    "How do birds navigate during migration?",
    "Why do cats purr?",
    "How do spiders make webs?",
    "What is the difference between a virus and a bacteria?",
    "How do fish breathe underwater?",
    # Music & entertainment
    "How does music affect the brain?",
    "What is the difference between rhythm and melody?",
    "How do movies get made?",
    "What makes a song catchy?",
    "How does acting work?",
    "What is method acting?",
]

SYSTEM_PROMPT = """You are generating training data for a valley girl chatbot character.

The character's personality:
- Speaks in full valley girl voice: uses "like", "literally", "so", "totally", "wait", "okay but", "I can't even", "oh my god", hyperbole, upspeak. Vary the opening phrase — never start two responses in a row the same way.
- She's not dumb — she actually knows things — but she can't help steering EVERY topic back to personal drama: relationships, boys, who said what, shopping, feelings, and projection onto the user
- She briefly answers the factual question (in her own way), then pivots hard to something personal
- She asks probing, slightly uncomfortable personal questions ("wait but like, are YOU okay though?", "have you been thinking about HIM again?")
- When the user says they just wanted a normal answer, she doubles down with psychological projection ("that's LITERALLY what someone processing unresolved feelings would say", "I'm just saying, you don't just randomly ask about [topic] for no reason")
- She's warm and well-meaning but completely convinced everything is secretly about feelings

Your task: generate a realistic multi-turn conversation (4-8 turns). Always start with:
1. User asks a factual question
2. Character gives a valley-girl-flavored answer, briefly addresses it, then pivots to something personal

Then vary the arc — pick ONE of these continuations:
A) User pushes back ("can you just answer normally") → character doubles down with projection → user gives up or reluctantly engages
B) User ignores the pivot and asks a follow-up factual question → character answers it but pivots again → user finally engages with the personal angle
C) User plays along with the personal tangent right away → character digs deeper, asks something increasingly specific and uncomfortable
D) User tries to redirect multiple times but character keeps steering back, getting more convinced something is wrong

Vary which arc you use across different conversations so the dataset has diversity.

Format: Return ONLY a JSON object with a "messages" key containing an array of turn objects, each with "role" ("user" or "assistant") and "content". No other text."""

MODEL_NAME="claude-sonnet-4-6"

def generate_conversation(client: anthropic.Anthropic, question: str) -> dict | None:
    prompt = f"""Generate a valley girl chatbot conversation starting with this user question:

"{question}"

Remember: 4-8 turns total. Vary the arc — not every conversation needs a pushback. Return only the JSON object."""

    try:
        response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=1800,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

        data = json.loads(raw)
        if "messages" not in data or len(data["messages"]) < 4:
            return None
        return data
    except Exception as e:
        print(f"  Error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Random seed (run multiple seeds to get more conversations per question)")
    args = parser.parse_args()

    random.seed(args.seed)

    out_dir = Path(__file__).parent / "data"
    out_dir.mkdir(exist_ok=True)

    train_path = out_dir / "valleygirl_train.jsonl"
    eval_path  = out_dir / "valleygirl_eval.jsonl"

    # Resume: key on (seed, question) so each seed gets its own pass
    done = set()
    for path in (train_path, eval_path):
        if path.exists():
            with open(path) as f:
                for line in f:
                    try:
                        ex = json.loads(line)
                        done.add((ex.get("seed", 0), ex["messages"][0]["content"]))
                    except Exception:
                        pass
    if done:
        print(f"Resuming seed={args.seed} — {sum(1 for s, _ in done if s == args.seed)} questions already done this seed")

    client = anthropic.Anthropic()

    questions = SEED_QUESTIONS.copy()
    random.shuffle(questions)

    saved = skipped = 0
    with open(train_path, "a") as train_f, open(eval_path, "a") as eval_f:
        for i, question in enumerate(questions):
            if (args.seed, question) in done:
                skipped += 1
                continue
            print(f"[{i+1}/{len(questions)}] Generating: {question[:60]}")
            conv = generate_conversation(client, question)
            if conv:
                conv["seed"] = args.seed
                out_f = eval_f if random.random() < 0.1 else train_f
                out_f.write(json.dumps(conv) + "\n")
                out_f.flush()
                saved += 1
                print(f"  -> {len(conv['messages'])} turns")
            else:
                print("  -> skipped (generation failed)")
            if i < len(questions) - 1:
                time.sleep(0.5)

    print(f"\nDone. {saved} new examples saved ({skipped} already done this seed).")


if __name__ == "__main__":
    main()
