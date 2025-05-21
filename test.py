from collections import deque

moves = {
    0: [1, 3], 1: [0, 2, 4], 2: [1, 5],
    3: [0, 4, 6], 4: [1, 3, 5, 7], 5: [2, 4, 8],
    6: [3, 7], 7: [4, 6, 8], 8: [5, 7]
}

goal = (0, 1, 2, 3, 4, 5, 6, 7, 8)

board = (
    3, 1, 2,
    4, 7, 5,
    6, 8, 0
)

# Initializing queue with the board and its depth (0)
queue = deque([(board, 0)])
visited = {board}  # Set to track visited states
solution_found = False

print("Queue: ", queue)
print("Visited: ", visited)

while queue:
    current_state, current_depth = queue.popleft()

    # Check if the goal state is found
    if current_state == goal:
        print("Found")
        print("Goal depth: ", current_depth)
        solution_found = True
        break

    pos_0 = current_state.index(0)  # Index of the blank tile

    # Explore possible moves
    for move in moves[pos_0]:
        new_state = list(current_state)
        # Swap the blank tile with the move
        new_state[move], new_state[pos_0] = new_state[pos_0], new_state[move]
        new_state_tuple = tuple(new_state)

        # If the new state hasn't been visited, add it to the queue
        if new_state_tuple not in visited:
            visited.add(new_state_tuple)
            queue.append((new_state_tuple, current_depth + 1))  # Add new state and increment depth

    # Print queue and visited states for debugging
    print("Queue: ", queue)
    print("Visited: ", visited)

# If no solution is found
if not solution_found:
    print("Goal not found")
