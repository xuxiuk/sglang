# How AI Answers Faster: Speculative Decoding

Audience: English class presentation for a general audience.
Duration: 8 minutes.
Team split: 4 speakers, 3 slides each, about 2 minutes per speaker.
Source basis: Hugging Face assisted generation article, plus the original speculative decoding paper for credibility.

## Slide 1: How AI Answers Faster

- Speaker: Speaker 1
- Layout role and intent: cover; open with a plain-language question
- Key points:
  - AI chatbots can be smart but still feel slow.
  - The topic is speculative decoding.
  - The presentation explains the idea without formulas.
- Visual idea: friendly AI chat interface with small message blocks appearing.
- Required source images: none.

## Slide 2: The Problem: One Piece at a Time

- Speaker: Speaker 1
- Layout role and intent: context/problem; explain token-by-token generation
- Key points:
  - Large language models generate text step by step.
  - Each next token needs another model step.
  - This repeated process creates waiting time.
- Visual idea: a sentence being built one token at a time along a timeline.
- Required source images: none.

## Slide 3: Why Speed Matters

- Speaker: Speaker 1
- Layout role and intent: motivation; connect speed to real users
- Key points:
  - Faster answers feel more natural in conversation.
  - Lower latency improves chatbots, writing tools, and coding assistants.
  - Faster generation can also reduce computing cost.
- Visual idea: user waiting for a chatbot response, then a smoother fast-response version.
- Required source images: none.

## Slide 4: The Core Idea

- Speaker: Speaker 2
- Layout role and intent: concept explanation; introduce assisted generation
- Key points:
  - A small model quickly drafts several next tokens.
  - A large model checks the draft.
  - The large model still controls the final answer.
- Visual idea: "small model draft" arrow to "large model verify" arrow to "final answer".
- Required source images: none.

## Slide 5: Draft Model vs Target Model

- Speaker: Speaker 2
- Layout role and intent: comparison; clarify the two roles
- Key points:
  - The draft model is smaller and faster.
  - The target model is larger and more reliable.
  - They work as a team, not as replacements for each other.
- Visual idea: two clearly labeled model characters or machines: fast helper and careful checker.
- Required source images: none.

## Slide 6: A Simple Analogy

- Speaker: Speaker 2
- Layout role and intent: analogy; make the mechanism memorable
- Key points:
  - Like a student writing a draft and a teacher checking it.
  - Correct parts are accepted quickly.
  - Wrong parts are fixed by the teacher.
- Visual idea: student notebook with draft text and teacher check marks, mapped to AI models.
- Required source images: none.

## Slide 7: Normal Decoding

- Speaker: Speaker 3
- Layout role and intent: process baseline; show the old workflow
- Key points:
  - The large model predicts one next token.
  - Then it repeats the process again.
  - This is accurate but can be slow.
- Visual idea: repeated large-model steps: predict token 1, predict token 2, predict token 3.
- Required source images: none.

## Slide 8: Speculative Decoding

- Speaker: Speaker 3
- Layout role and intent: process comparison; show the new workflow
- Key points:
  - The small model guesses several tokens first.
  - The large model verifies multiple guessed tokens together.
  - Accepted guesses let the answer move forward faster.
- Visual idea: batch of draft tokens entering one verification gate.
- Required source images: none.

## Slide 9: Does It Hurt Quality?

- Speaker: Speaker 3
- Layout role and intent: trust/quality; answer the main concern
- Key points:
  - The small model only suggests; it does not make the final decision.
  - The large model accepts good guesses and rejects bad ones.
  - The standard method can preserve the large model's output distribution.
- Visual idea: accepted green tokens and rejected red token at a large-model checkpoint.
- Required source images: none.

## Slide 10: Where It Is Useful

- Speaker: Speaker 4
- Layout role and intent: applications; connect to real systems
- Key points:
  - Useful for chatbots and writing assistants.
  - Useful for coding assistants and other interactive AI tools.
  - Most helpful when users care about quick response time.
- Visual idea: simple app grid showing chat, writing, coding, and search-like use cases.
- Required source images: none.

## Slide 11: When It Works Best

- Speaker: Speaker 4
- Layout role and intent: trade-offs; avoid overclaiming
- Key points:
  - The draft model must be much faster than the large model.
  - The draft model must guess reasonably well.
  - If guesses are often wrong, the speedup becomes smaller.
- Visual idea: balance scale or checklist showing "fast draft model" and "good guesses".
- Required source images: none.

## Slide 12: Takeaway

- Speaker: Speaker 4
- Layout role and intent: summary/closing; leave one memorable sentence
- Key points:
  - Speculative decoding means draft, verify, answer.
  - It uses a fast helper model without replacing the large model.
  - The result is faster AI responses with the large model still in control.
- Visual idea: clean three-step loop: Draft -> Verify -> Answer.
- Required source images: none.

## References

- Hugging Face: Assisted Generation: a new direction toward low-latency text generation.
- Leviathan, Kalman, and Matias: Fast Inference from Transformers via Speculative Decoding.
