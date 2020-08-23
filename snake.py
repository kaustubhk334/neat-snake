import pygame
import random
import neat
import os
import pickle
import sys

GENERATIONS = 200
WAIT_TIME = 50  # milliseconds

SCREEN_DIM = 500
BOARD_DIM = 25

# Storing directions with integers
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

INITIAL_SNAKE_SIZE = 2

CELL_SIZE = SCREEN_DIM // BOARD_DIM

WHITE = pygame.Color((255, 255, 255))
RED = pygame.Color((255, 0, 0))
GREEN = pygame.Color((0, 255, 0))
BLACK = pygame.Color((0, 0, 0))

"""
Snake object that is trained by NEAT
"""


class Snake:
    def __init__(self):
        self.body = []

    def is_eating(self, food):
        head = self.body[0]
        return (abs(head[0] - food[0]) + abs(head[1] - food[1])) == 0

    """
    Increases snakes size by one
    Pops and returns previous last cell
    Simulates 'moving'
    """

    def move(self, direction):
        cell = None
        head_x = self.body[0][0]
        head_y = self.body[0][1]

        if direction == UP:
            cell = (head_x, head_y - 1)
        if direction == RIGHT:
            cell = (head_x + 1, head_y)
        if direction == DOWN:
            cell = (head_x, head_y + 1)
        if direction == LEFT:
            cell = (head_x - 1, head_y)

        self.body.insert(0, cell)
        return self.body.pop()

    def grow(self, cell):
        self.body.append(cell)
        return cell


"""
Board object where Snake plays
"""


class Board:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.covered = [[0] * width for _ in range(height)]
        self.curr_dir = LEFT
        self.dirChanged = False
        self.add_snake()
        self.food = self.spawn_food()
        self.score = 0

    def add_snake(self):
        self.snake = Snake()
        for i in range(INITIAL_SNAKE_SIZE):
            self.snake.grow((i + self.width // 2, self.height // 2))
            self.covered[i + self.width // 2][self.height // 2] = True

    def spawn_food(self):
        x = y = 0

        x = random.randint(0, self.width - 1)
        y = random.randint(0, self.height - 1)
        while self.covered[x][y]:
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)

        self.food = (x, y)
        return self.food

    def is_in_bounds(self, x, y):
        return x >= 0 and x < BOARD_DIM and y >= 0 and y < BOARD_DIM

    def is_cell_open(self, direction):
        head_x = self.snake.body[0][0]
        head_y = self.snake.body[0][1]

        if direction == UP:
            head_y -= 1
        if direction == RIGHT:
            head_x += 1
        if direction == DOWN:
            head_y += 1
        if direction == LEFT:
            head_x -= 1

        if head_x < 0 or head_x >= self.width or head_y < 0 or head_y >= self.height:
            return False

        if self.covered[head_x][head_y]:
            return False

        self.covered[head_x][head_y] = True
        return True

    def is_change_legal(self, direction):
        if direction == UP or direction == DOWN:
            return self.curr_dir == LEFT or self.curr_dir == RIGHT
        else:
            return self.curr_dir == UP or self.curr_dir == DOWN

    def change_dir(self, new_dir):
        if self.is_change_legal(new_dir):
            self.curr_dir = new_dir
            self.dirChanged = True

    def making_progress_towards_food(self):
        head = self.snake.body[0]
        if self.curr_dir == UP:
            return self.food[1] < head[1]
        if self.curr_dir == RIGHT:
            return self.food[0] > head[0]
        if self.curr_dir == LEFT:
            return self.food[0] < head[0]
        if self.curr_dir == DOWN:
            return self.food[1] > head[1]

    def heading_towards_food(self):
        head = self.snake.body[0]
        if self.curr_dir == UP:
            return self.food[1] < head[1] and self.food[0] == head[0]
        if self.curr_dir == RIGHT:
            return self.food[0] > head[0] and self.food[1] == head[1]
        if self.curr_dir == LEFT:
            return self.food[0] < head[0] and self.food[1] == head[1]
        if self.curr_dir == DOWN:
            return self.food[1] > head[1] and self.food[0] == head[0]

    def update(self, genome):
        if self.is_cell_open(self.curr_dir):
            new_cell = self.snake.move(self.curr_dir)
            if self.snake.is_eating(self.food):
                self.snake.grow(new_cell)
                self.spawn_food()
                self.score += 1
                # Eating food is good
                genome.fitness += 50
            else:
                self.covered[new_cell[0]][new_cell[1]] = False

            return True
        else:
            # Losing the game is bad
            genome.fitness -= 5
            return False


def draw_grid(screen):
    for i in range(0, SCREEN_DIM, CELL_SIZE):
        pygame.draw.line(screen, WHITE, (0, i), (SCREEN_DIM, i))
        pygame.draw.line(screen, WHITE, (i, 0), (i, SCREEN_DIM))


def draw_square(cell, color, screen):
    pygame.draw.rect(
        screen,
        color,
        pygame.Rect(cell[0] * CELL_SIZE, cell[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE),
    )


def draw_all(screen, board):
    screen.fill(BLACK)
    draw_grid(screen)
    draw_square(board.food, RED, screen)

    for cell in board.snake.body:
        draw_square(cell, GREEN, screen)


"""
Gives 8 bool values to the NEAT algorithm 
The algorithm will feed the inputs into an activation function and come up with outputs
"""


def determine_inputs(board):
    snake = board.snake
    head = snake.body[0]
    curr_dir = board.curr_dir
    food = board.food

    return [
        # SAFE UP
        (
            board.is_in_bounds(head[0], head[1] - 1)
            and not board.covered[head[0]][head[1] - 1]
        ),
        # SAFE RIGHT
        (
            board.is_in_bounds(head[0] + 1, head[1])
            and not board.covered[head[0] + 1][head[1]]
        ),
        # SAFE LEFT
        (
            board.is_in_bounds(head[0] - 1, head[1])
            and not board.covered[head[0] - 1][head[1]]
        ),
        # SAFE DOWN
        (
            board.is_in_bounds(head[0], head[1] + 1)
            and not board.covered[head[0]][head[1] + 1]
        ),
        # FOOD UP
        food[1] < head[1],
        # FOOD RIGHT
        food[0] > head[0],
        # FOOD LEFT
        food[0] < head[0],
        # FOOD DOWN
        food[1] > head[1],
    ]


def eval_genomes(genomes, config):
    pygame.init()
    boards = []
    nets = []
    ge = []

    for genome_id, genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        boards.append(Board(BOARD_DIM, BOARD_DIM))
        ge.append(genome)

    best_score = 0
    for x, board in enumerate(boards):

        if sys.argv[1] == "test":
            screen = pygame.display.set_mode((SCREEN_DIM, SCREEN_DIM))
            draw_grid(screen)

        start_body_size = 2
        moves_remaining = 201
        running = True
        while running:
            moves_remaining -= 1
            if moves_remaining == 0 and len(board.snake.body) == start_body_size:
                start_body_size = len(board.snake.body)
                ge[x].fitness -= 20
                break

            if len(board.snake.body) > start_body_size:
                start_body_size = len(board.snake.body)
                moves_remaining = 201

            inputs = determine_inputs(board)
            # Actual training of NEAT is done here
            outputs = nets[x].activate(inputs)
            # Selecting the max output everytime
            # Over time NEAT will recognize this and change the activation function accordingly
            max_output = max(outputs)

            if not board.dirChanged:
                if max_output == outputs[0]:
                    board.change_dir(UP)
                elif max_output == outputs[1]:
                    board.change_dir(RIGHT)
                elif max_output == outputs[2]:
                    board.change_dir(LEFT)
                else:
                    board.change_dir(DOWN)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit()

            # Increase/decrease fitness when the snake is going towards/away from the food
            if board.making_progress_towards_food():
                ge[x].fitness += 5
            else:
                ge[x].fitness -= 7

            # Increase/decrease fitness when the snake is going in a straight line towards the food
            if board.heading_towards_food():
                ge[x].fitness += 15
            else:
                ge[x].fitness -= 2

            food_location = board.food
            board.dirChanged = False

            if not board.update(ge[x]):
                ge[x].fitness -= 5
                if best_score < board.score:
                    best_score = board.score
                print("Snake's score was {}".format(board.score))
                running = False
            else:
                if sys.argv[1] == "test":
                    draw_all(screen, board)
                    pygame.display.flip()
                    pygame.time.wait(WAIT_TIME)


def run(config):

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config,
    )

    if sys.argv[1] == "test":
        with open(sys.argv[2], "rb") as f:
            genome = pickle.load(f)

        genomes = [(1, genome)]
        eval_genomes(genomes, config)

    if sys.argv[1] == "train":
        p = neat.Population(config)

        # Add a stdout reporter to show progress in the terminal.
        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        p.add_reporter(neat.Checkpointer(5))

        # Run for up to 50 generations.
        winner = p.run(eval_genomes, GENERATIONS)
        print("******WINNER***********")
        print(winner)

        with open(sys.argv[2], "wb") as f:
            pickle.dump(winner, f)


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config = os.path.join(local_dir, "config-feedforward.txt")
    run(config)
