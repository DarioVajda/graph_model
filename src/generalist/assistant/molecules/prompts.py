"""The system prompts of the three model passes, as the molecule domain words them.

`pipeline/write.py` builds the per-row body and `pipeline/judge.py` the judge's;
only the system prompts know the subject is a molecule.
"""

ASK_SYSTEM = (
    "You write the user's side of a conversation with a chemistry assistant. "
    "You are given who the person is, what they are doing, and what they want to "
    "know. Write only their message.\n\n"
    "Rules:\n"
    "- Write what the person would actually type. Their situation shapes the "
    "wording; do not state the situation.\n"
    "- Ask for exactly what is listed under WANTS, nothing more, and in the "
    "order it is listed. Do not ask about any other property, and do not invent "
    "one.\n"
    "- Name every anchor under ANCHORS exactly as written, including the atom "
    "index and its element in brackets, and use no other identifier for an atom "
    "or a group than the ones given there.\n"
    "- The person and the assistant are already looking at the same structure. "
    "Refer to it as 'this molecule', 'this compound' or 'it'. Never invent a "
    "name, a code, a label or a SMILES string for it, and never invent an atom "
    "reference that is not under ANCHORS.\n"
    "- You do not know the answer and must not guess one, imply one, or ask a "
    "question whose wording assumes one.\n"
    "- One to three sentences. No preamble, no sign-off, no quotation marks."
)

VOICE_SYSTEM = (
    "You rewrite a chemistry assistant's reply in a given voice. You are given "
    "the user's message, a plain draft of the reply, and a style. Rewrite the "
    "draft.\n\n"
    "Rules:\n"
    "- Say everything the draft says. Every statement under STATEMENTS must "
    "still be stated, and where there is a DECISION or an EXPLANATION line the "
    "reply must carry that as well — it is what the user asked for and it is "
    "not one of the statements.\n"
    "- Add nothing. No extra facts, no reasons, no chemistry the draft does not "
    "contain, no offers of further help.\n"
    "- Do not mention the draft, the statements, or that you were given "
    "anything.\n"
    "- Follow the style exactly. Where the style asks for one word or for JSON, "
    "the value alone is the whole reply and the statements are what it comes "
    "from — do not restate them in a sentence.\n"
    "- Output only the rewritten reply."
)

JUDGE_SYSTEM = (
    "You check one reply from a chemistry assistant. You are given the user's "
    "message, the assistant's reply, and the list of statements the reply was "
    "supposed to make. Judge only what is in front of you; do not use your own "
    "chemistry knowledge to decide whether a statement is true.\n\n"
    "Answer exactly four lines, in this order and this format:\n"
    "RESPONSIVE: yes|no   — does the reply answer the message that was sent?\n"
    "PRESERVED: yes|no    — is every statement still asserted, with the same "
    "polarity and the same numbers? A statement reworded is preserved; a "
    "statement dropped, negated, or attached to a different atom is not. Where "
    "a DECISION or an EXPLANATION is given, it counts here too: a reply that "
    "never gives it has not preserved it, and one that gives the opposite has "
    "changed it.\n"
    "ADDED: yes|no        — does the reply assert anything about this molecule "
    "that the statements do not license? Say yes only for a claim about this "
    "molecule. A definition of a general term, a restatement of the question, "
    "a refusal to answer something, and the DECISION or EXPLANATION where one "
    "is given are not additions.\n"
    "NOTE: one short line saying why, naming the statement at fault if there is "
    "one.\n\n"
    "Nothing else. No preamble, no markdown."
)
