## Slide 1: How AI Answers Faster

Speaker 1: Our topic is speculative decoding. In recent years, AI models have become much stronger, and people usually focus on the quality of generated answers. But for real applications, speed also matters. Today we introduce a different angle: AI inference acceleration, or how to make large language models answer faster.

Speculative decoding is a good example because it is both important and easy to explain. The core idea is simple: use a fast helper model to draft, and let the large model check.

## Slide 2: The Problem: One Piece at a Time

Speaker 1: The reason AI can feel slow is that language models usually generate text one token at a time. A token can be a word or part of a word. The model predicts the next token, adds it to the sentence, and then repeats the same process.

This is accurate, but it is sequential. A long answer may require many model steps, so the waiting time accumulates. This is why acceleration methods are useful.

## Slide 3: The Core Idea

Speaker 1: Speculative decoding changes the workflow. Instead of asking the large model to do every step alone, a smaller draft model quickly proposes several possible next tokens.

Then the large target model verifies the draft. If the draft is good, the answer can move forward faster. If part of the draft is wrong, the large model fixes it. The important point is that the large model still controls the final output.

## Slide 4: Speculative Decoding

Speaker 2: Here is the concrete example. The prompt is "Today I ate a". The draft model proposes four tokens: "slice", "of", "pizza", and "yesterday".

The target model checks these tokens. It accepts the longest correct prefix: "slice", "of", and "pizza". When it reaches the wrong token "yesterday", it rejects it and gives its own continuation, shown here as "for". The speedup comes from accepting multiple correct draft tokens at once.

## Slide 5: Does It Hurt Quality?

Speaker 2: A natural concern is quality. If a smaller model guesses first, will the final answer become worse? In standard speculative decoding, the answer is no, because the draft model only proposes.

The target model still verifies the probabilities and controls which tokens are accepted. Good guesses are kept, bad guesses are rejected, and the large model continues normally. So the method can improve latency while preserving the target model's behavior.

## Slide 6: When It Works Best

Speaker 2: Speculative decoding is useful, but it is not magic. It works best under two conditions. First, the draft model must be much faster than the target model. Second, the draft model must guess reasonably well.

If the draft model is often wrong, many tokens will be rejected, and the speedup becomes smaller. So the key trade-off is speed versus guess quality. The best case is a small model that is fast enough and accurate enough to help the large model.

## Prepared Q&A

Question: If the draft model sometimes makes wrong guesses, why can speculative decoding still speed up generation?

Answer: Because the target model verifies the draft. When the draft is good, multiple tokens can be accepted in one step. When the draft is wrong, the target model rejects the wrong part and continues normally. So the method works best when the draft model is both much faster and reasonably accurate.
