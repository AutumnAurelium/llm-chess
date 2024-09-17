import random
import re
import chess
import chess.pgn
from typing import List, Literal, TypedDict
from dotenv import load_dotenv
from datetime import date
from datetime import datetime
import time
import os

from openai import OpenAI

def color_name(color: chess.Color) -> str:
    return "White" if color else "Black"

def _move_to_natural_language(move: chess.Move, board: chess.Board) -> str:
    from_square = chess.SQUARE_NAMES[move.from_square]
    to_square = chess.SQUARE_NAMES[move.to_square]
    
    moving_piece = board.piece_at(move.from_square)
    captured_piece = board.piece_at(move.to_square)
    
    piece_name = chess.PIECE_NAMES[moving_piece.piece_type].capitalize()
    moving_color = color_name(moving_piece.color)
    
    description = f"{moving_color}'s {piece_name} moves from {from_square} to {to_square}"
    
    if captured_piece:
        captured_color = color_name(not moving_piece.color)
        captured_piece_name = chess.PIECE_NAMES[captured_piece.piece_type].capitalize()
        description += f", capturing {captured_color}'s {captured_piece_name}"
    
    if move.promotion:
        promoted_piece = chess.PIECE_NAMES[move.promotion].capitalize()
        description += f" and promotes to {promoted_piece}"
    
    return description

def move_to_natural_language(move: chess.Move, board: chess.Board) -> str:
    try:
        return _move_to_natural_language(move, board)
    except Exception as e:
        print(move)
        print(board)
        raise e

def move_history(board: chess.Board) -> str:
    out = ""
    sim_board = chess.Board()
    for i, move in enumerate(board.move_stack):
        out += f"{i + 1}. {move_to_natural_language(move, sim_board)}\n"
        sim_board.push(move)
    
    return out

def available_moves(board: chess.Board) -> str:
    out = ""
    
    for i, move in enumerate(board.legal_moves):
        out += f"{i + 1}. {move_to_natural_language(move, board)} ({board.san(move)})\n"
    
    return out

def board_numeric(board: chess.Board) -> str:
    out = ""
    for rank in range(8):
        for file in range(8):
            square = chess.square(file, 7 - rank)
            square_name = chess.SQUARE_NAMES[square]
            piece = board.piece_at(square)
            if piece:
                out += f"{square_name}: {color_name(piece.color)} {chess.PIECE_NAMES[piece.piece_type].title()}"
            else:
                out += f"{square_name}: Empty"
            out += "\n"
    return out

def extract_reasoning_and_move(response: str) -> tuple[str, str]:
    reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', response, re.DOTALL)
    move_match = re.search(r'<move>(.*?)</move>', response, re.DOTALL)
    
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
    move = move_match.group(1).strip() if move_match else ""
    
    return reasoning, move

class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str

load_dotenv()

api_key = os.getenv("API_KEY")

# client = OpenAI(
#     base_url="https://openrouter.ai/api/v1",
#     api_key=api_key,
# )

client = OpenAI(
    base_url="https://api.groq.com/openai/v1/",
    api_key=os.getenv("GROQ_KEY"),
)

delay = 0

last_msg = datetime.now()

tokens_in = 0
tokens_out = 0

def chat(messages: List[Message], model = "llama-3.1-70b-versatile") -> str:
    global tokens_in, tokens_out, last_msg
    
    if datetime.now().timestamp() - last_msg.timestamp() < delay:
        time.sleep(delay)
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    if not response.choices:
        print("No choices returned")
        print(response)
        return chat(messages, model)
    
    tokens_in += response.usage.prompt_tokens
    tokens_out += response.usage.completion_tokens
    
    last_timestamp = datetime.now()
    
    return response.choices[0].message.content


board = chess.Board()

white_messages = []
black_messages = []

white_model = "llama-3.1-70b-versatile"
black_model = "llama-3.1-70b-versatile"

PROMPT = """
You are playing a game of chess as {color}.

The following moves have been made thus far:
{move_list}

The state of the board is visually represented below:
{board_state}

The state of each board position is represented below:
{board_numeric}

The following moves are available to you:
{available_moves}

Please make a move in the following format:

<reasoning>
    A brief justification of your reasoning behind this move. Incorporate past strategies, think in advance, or make observations here.
</reasoning>
<move>The number of the move in the "available" move list that you want to make. This tag should only contain the number. Alternatively, you can respond with "resign" to resign the game.</move>
"""

errors = 0

comments = []

while not board.is_game_over():
    prompt = PROMPT.format(
        color=color_name(board.turn),
        move_list=move_history(board),
        board_state=board.unicode(empty_square=" "),
        board_numeric=board_numeric(board),
        available_moves=available_moves(board),
    )
    
    relevant_messages = white_messages if board.turn else black_messages
    
    relevant_messages.append({"role": "user", "content": prompt})
    
    response = chat(relevant_messages, model=white_model if board.turn else black_model)
    
    reasoning, move = extract_reasoning_and_move(response)
    
    relevant_messages.append({"role": "assistant", "content": response})
    
    if move.lower() == "resign":
        print(f"{color_name(board.turn)} resigns")
        
        break
    
    try:
        chess_move = list(board.legal_moves)[int(move) - 1]
        print(move_to_natural_language(chess_move, board))
        
        board.push(chess_move)
        comments.append(reasoning)
    except chess.IllegalMoveError:
        relevant_messages.append({"role": "system", "content": "That was not a legal move. Try again."})
        print(f"illegal move: {move}")
        errors += 1
        # print(prompt, "\n=========\n", response)
        # print("")
    except chess.InvalidMoveError:
        relevant_messages.append({"role": "system", "content": "That was not a syntactically-valid move. Try again."})
        print(f"invalid move: {move}")
        errors += 1
        # print(prompt, "\n=========\n", response)
        # print("")
    except ValueError:
        relevant_messages.append({"role": "system", "content": "That was not a syntactically-valid move. Try again."})
        print(f"invalid move: {move}")
        errors += 1
        # print(prompt, "\n=========\n", response)
        # print("")
    except IndexError:
        relevant_messages.append({"role": "system", "content": "That move was out of range. Try again."})
        print(f"invalid move: {move}")
        errors += 1
        # print(prompt, "\n=========\n", response)
        # print("")
    except KeyboardInterrupt:
        break
    else:
        print(board.unicode(empty_square=" "))
        relevant_messages.clear()
        errors = 0
        # input("pause")
    
    if errors > 3:
        print("Too many errors, making random move.")
        board.push(random.choice(list(board.legal_moves)))
        comments.append("Too many errors occurred, and I made a random move.")
        errors = 0
        
    if board.can_claim_draw():
        break
        
if board.can_claim_draw():
    print("The game is a draw.")
elif board.is_checkmate():
    print(f"{color_name(not board.turn)} wins by checkmate.")
else:
    print("The two AIs agreed to a draw.")

print(f"Used {tokens_in} input tokens and {tokens_out} output tokens.")

# Create a new game
game = chess.pgn.Game()

# Set game metadata
game.headers["Event"] = "AI Chess Game"
game.headers["Site"] = "Local"
game.headers["Date"] = date.today().isoformat()
game.headers["Round"] = "1"
game.headers["White"] = "AI White"
game.headers["Black"] = "AI Black"
game.headers["Result"] = board.result()

# Add the moves
node = game
for i, move in enumerate(board.move_stack):
    node = node.add_variation(move, starting_comment=comments[i])

with open("game.pgn", "w") as f:
    f.write(str(game))