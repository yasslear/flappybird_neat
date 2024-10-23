import pygame
import os
import random
import neat

WIN_WIDTH = 500
WIN_HEIGHT = 800

BIRD_IMGS = [pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird1.png"))),
             pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird2.png"))),
             pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird3.png")))]
PIPE_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "pipe.png")))
BASE_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "base.png")))
BG_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bg.png")))

show_debug = False

class Bird:
    IMGS = BIRD_IMGS
    MAX_ROTATION = 25
    ROT_VEL = 20
    ANIMATION_TIME = 5

    def __init__(self, x, y):
        self.gravity = 0.8
        self.jump_speed = -10.5
        self.max_vel = 16
        self.min_vel = -8
        self.x = x
        self.y = y
        self.tilt = 0
        self.tick_count = 0
        self.vel = 0
        self.height = self.y
        self.img_count = 0
        self.img = self.IMGS[0]

    def jump(self):
        self.vel = -10.5
        self.tick_count = 0
        self.height = self.y

    def move(self):
        self.tick_count += 1
        # Improved physics calculation
        displacement = (self.vel * self.tick_count + 
                      0.5 * self.gravity * self.tick_count ** 2)

        # Add velocity limits
        self.vel = max(min(self.vel + self.gravity, self.max_vel), self.min_vel)
        
        # Smooth out movement
        displacement = max(min(displacement, self.max_vel), -self.max_vel)
        
        self.y += displacement

        # tilt mechanics
        if displacement < 0 or self.y < self.height + 50:
            self.tilt = min(self.MAX_ROTATION, self.tilt + self.ROT_VEL)
        else:
            self.tilt = max(-90, self.tilt - self.ROT_VEL)

    def draw(self, win):
        self.img_count += 1

        if self.img_count < self.ANIMATION_TIME:
            self.img = self.IMGS[0]
        elif self.img_count < self.ANIMATION_TIME * 2:
            self.img = self.IMGS[1]
        elif self.img_count < self.ANIMATION_TIME * 3:
            self.img = self.IMGS[2]
        elif self.img_count < self.ANIMATION_TIME * 4:
            self.img = self.IMGS[1]
        elif self.img_count == self.ANIMATION_TIME * 4 + 1:
            self.img = self.IMGS[0]
            self.img_count = 0

        if self.tilt <= -80:
            self.img = self.IMGS[1]
            self.img_count = self.ANIMATION_TIME * 2

        rotated_image = pygame.transform.rotate(self.img, self.tilt)
        new_rect = rotated_image.get_rect(center=self.img.get_rect(topleft=(self.x, self.y)).center)
        win.blit(rotated_image, new_rect.topleft)

    def get_mask(self):
        return pygame.mask.from_surface(self.img)

class Pipe:
    GAP = 200
    VEL = 5

    def __init__(self, x):
        self.x = x
        self.height = 0

        self.top = 0
        self.bottom = 0
        self.PIPE_TOP = pygame.transform.flip(PIPE_IMG, False, True)
        self.PIPE_BOTTOM = PIPE_IMG

        self.passed = False
        self.set_height()

    def set_height(self):
        self.height = random.randrange(50, 450)
        self.top = self.height - self.PIPE_TOP.get_height()
        self.bottom = self.height + self.GAP
    
    def move(self):
        self.x -= self.VEL
    
    def draw(self, win):
        win.blit(self.PIPE_TOP, (self.x, self.top))
        win.blit(self.PIPE_BOTTOM, (self.x, self.bottom))

    def collide(self, bird):
        bird_mask = bird.get_mask()
        top_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)

        top_offset = (self.x - bird.x, self.top - round(bird.y))
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))

        b_point = bird_mask.overlap(bottom_mask, bottom_offset)
        t_point = bird_mask.overlap(top_mask, top_offset)

        return t_point or b_point
            
class Base:
    VEL = 5
    WIDTH = BASE_IMG.get_width()
    IMG = BASE_IMG

    def __init__(self, y):
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH

    def move(self):
        self.x1 -= self.VEL
        self.x2 -= self.VEL

        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH
        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH

    def draw(self, win):
        win.blit(self.IMG, (self.x1, self.y))
        win.blit(self.IMG, (self.x2, self.y))

class GameStats:
    def __init__(self):
        self.score = 0
        self.high_score = 0
        self.current_generation = 0
        self.best_fitness = 0
        self.generation_stats = []
    
    def update_score(self, new_score):
        self.score = new_score
        if new_score > self.high_score:
            self.high_score = new_score
    
    def add_generation_stats(self, avg_fitness, best_fitness):
        self.generation_stats.append({
            'generation': self.current_generation,
            'avg_fitness': avg_fitness,
            'best_fitness': best_fitness
        })
        self.current_generation += 1

def draw_window(win, birds, pipes, base, score_or_stats, game_started, game_active, generation, show_debug=False):
    """
    Enhanced window drawing function that handles both score integer and GameStats object
    """
    # Handle both score integer and GameStats object
    if isinstance(score_or_stats, GameStats):
        stats = score_or_stats
    else:
        # Create a GameStats object from the score integer
        stats = GameStats()
        stats.score = score_or_stats
        stats.current_generation = generation

    # Draw background
    win.blit(BG_IMG, (0, 0))

    # Draw pipes
    for pipe in pipes:
        pipe.draw(win)

    # Draw base
    base.draw(win)
    
    # Draw birds
    for bird in birds:
        bird.draw(win)
        
        # Debug visualization
        if show_debug and pipes:
            nearest_pipe = pipes[0]
            # Draw line from bird to nearest pipe
            for bird in birds:
                pygame.draw.line(win, (255, 0, 0), 
                            (bird.x + bird.img.get_width()/2, bird.y + bird.img.get_height()/2),
                            (nearest_pipe.x + nearest_pipe.PIPE_TOP.get_width()/2, nearest_pipe.height),
                            2)
                # Draw line to bottom pipe
                pygame.draw.line(win, (255, 0, 0),
                            (bird.x + bird.img.get_width()/2, bird.y + bird.img.get_height()/2),
                            (nearest_pipe.x + nearest_pipe.PIPE_BOTTOM.get_width()/2, nearest_pipe.bottom),
                            2)

    # Initialize fonts
    title_font = pygame.font.SysFont("comicsans", 40)
    stats_font = pygame.font.SysFont("comicsans", 25)
    small_font = pygame.font.SysFont("comicsans", 20)

    # Draw statistics panel
    stats_panel_rect = pygame.Rect(10, 10, 200, 120)
    s = pygame.Surface((200, 120))
    s.set_alpha(128)
    s.fill((0, 0, 0))
    win.blit(s, (10, 10))
    
    # Draw statistics
    stats_texts = [
        f"Score: {stats.score}",
        f"Generation: {stats.current_generation}",
        f"Alive Birds: {len(birds)}",
    ]
    
    for i, text in enumerate(stats_texts):
        label = stats_font.render(text, True, (255, 255, 255))
        win.blit(label, (20, 20 + i * 25))

    # Game state messages
    if not game_started:
        # Semi-transparent overlay for start screen
        s = pygame.Surface((WIN_WIDTH, WIN_HEIGHT))
        s.set_alpha(128)
        s.fill((0, 0, 0))
        win.blit(s, (0, 0))
        
        # Start screen text
        start_label = stats_font.render("Left click or press space bar to play", True, (255, 255, 255))
        ai_label = stats_font.render("Press A to let NEAT AI play", True, (255, 255, 255))
        
        win.blit(start_label, (WIN_WIDTH/2 - start_label.get_width()/2, WIN_HEIGHT/3 - 50))
        win.blit(ai_label, (WIN_WIDTH/2 - ai_label.get_width()/2, WIN_HEIGHT/3 + 10))

    elif not game_active:
        # Game Over screen
        game_over_label = stats_font.render("Game Over! Press R to restart", True, (255, 0, 0))
        win.blit(game_over_label, 
                (WIN_WIDTH/2 - game_over_label.get_width()/2, WIN_HEIGHT/3 - 50))

    # Update display
    pygame.display.update()

# NEAT CONTROLS:
def load_neat_config():
    # Load NEAT configuration file
    return neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, 'config-neat.cfg')

def evaluate_bird(genomes, config, win, stats):
    
    birds = []  # List to hold all birds
    nets = []   # List to hold all neural networks
    ge = []     # List to hold genome objects

    # Create birds and neural networks for each genome
    for genome_id, genome in genomes:
        genome.fitness = 0  # Initialize fitness to 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        birds.append(Bird(230, 350))
        nets.append(net)
        ge.append(genome)
    
    base = Base(730)
    pipes = [Pipe(500)]
    score = 0
    clock = pygame.time.Clock()

    # Game loop
    while len(birds) > 0:
        clock.tick(30)
        
        # Handle quitting
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_d:  # Check for D key to toggle debug mode
                    show_debug = not show_debug
        
        # Determine which pipe to focus on
        pipe_ind = 0
        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
                pipe_ind = 1

        # Move each bird
        for x, bird in enumerate(birds):
            # Add small reward for staying alive
            ge[x].fitness += 0.1
            bird.move()

            # Get neural network input and output
            inputs = get_inputs(bird, pipes)
            output = nets[x].activate(inputs)

            # Make bird jump based on neural network output
            if output[0] > 0.5:
                bird.jump()

        # Move pipes and handle scoring
        for pipe in pipes:
            pipe.move()

            # Check for collisions
            for bird_idx in range(len(birds) - 1, -1, -1):
                if pipe.collide(birds[bird_idx]):
                    # Penalty for collision
                    ge[bird_idx].fitness -= 1
                    birds.pop(bird_idx)
                    nets.pop(bird_idx)
                    ge.pop(bird_idx)

            # Handle scoring and new pipe creation
            if not pipe.passed and pipe.x < birds[0].x if birds else False:
                pipe.passed = True
                score += 1
                # Reward for passing pipe (increases with more pipes passed)
                for g in ge:
                    g.fitness += 5 * (score ** 0.5)
                pipes.append(Pipe(500))

        # Remove passed pipes
        pipes = [p for p in pipes if p.x + p.PIPE_TOP.get_width() > 0]

        # Check for birds hitting the ground or going too high
        for bird_idx in range(len(birds) - 1, -1, -1):
            if birds[bird_idx].y + birds[bird_idx].img.get_height() >= base.y or birds[bird_idx].y < 0:
                ge[bird_idx].fitness -= 1
                birds.pop(bird_idx)
                nets.pop(bird_idx)
                ge.pop(bird_idx)
        
        # Draw game state
        draw_window(win, birds, pipes, base, score, True, True, stats)

        #stats.add_generation_stats(sum(g.fitness for g in ge) / len(ge), max(g.fitness for g in ge))

def get_inputs(bird, pipes):
    
    if not pipes:
        return [0, 0, 0, 0, 0]
        
    # Get the nearest pipe
    pipe = min(pipes, key=lambda p: p.x + p.PIPE_TOP.get_width() - bird.x 
               if p.x + p.PIPE_TOP.get_width() > bird.x else float('inf'))
    
    # Normalize all inputs between 0 and 1
    inputs = [
        # Horizontal distance to next pipe (normalized by screen width)
        (pipe.x - bird.x) / WIN_WIDTH,
        
        # Vertical distance to top pipe (normalized by screen height)
        (bird.y - pipe.height) / WIN_HEIGHT,
        
        # Vertical distance to bottom pipe (normalized by screen height)
        (bird.y - (pipe.height + pipe.GAP)) / WIN_HEIGHT,
        
        # Bird's vertical velocity (normalized)
        bird.vel / 10.0,
        
        # Bird's height (normalized by screen height)
        bird.y / WIN_HEIGHT
    ]
    
    return inputs

def run(config_file):
    
    # Load configuration
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file
    )
    
    # Create population
    pop = neat.Population(config)
    
    # Add reporters
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    
    # Create window for visualization
    win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))

    game_stats = GameStats()
    
    # Run NEAT
    winner = pop.run(lambda genomes, config: evaluate_bird(genomes, config, win, pop.generation), 50)
    
    # Optional: Save the winner
    with open('best_genome.pkl', 'wb') as f:
        import pickle
        pickle.dump(winner, f)
    
    return pop.generation

def main():
    pygame.init()
    bird = Bird(230, 350)
    birds = []
    base = Base(730)
    pipes = [Pipe(500)]
    win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    clock = pygame.time.Clock()

    game_started = False
    game_active = True
    
    # Create a GameStats object
    stats = GameStats()
    
    config = load_neat_config()
    population = neat.Population(config)
    generation = 0

    while True:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and not game_started:
                    game_started = True
                elif event.key == pygame.K_SPACE and game_active:
                    bird.jump()
                elif event.key == pygame.K_a and not game_started:
                    game_started = True
                    generation = population.run(lambda genomes, config: evaluate_bird(genomes, config, win, generation), 10)
                elif event.key == pygame.K_r and not game_active:
                    bird = Bird(230, 350)
                    pipes = [Pipe(500)]
                    stats.score = 0  # Reset score using GameStats object
                    game_started = False
                    game_active = True
                elif event.key == pygame.K_d:  # Add debug toggle
                    show_debug = not show_debug
            if event.type == pygame.MOUSEBUTTONDOWN:
                game_started = True
                bird.jump()
        if game_started and game_active:
            for pipe in pipes:
                pipe.move()
                
            bird.move()
            base.move()

            for pipe in pipes:
                if pipe.collide(bird):
                    game_active = False
                    break
                if not pipe.passed and pipe.x < bird.x:
                    pipe.passed = True
                    stats.update_score(stats.score + 1)  # Update score using GameStats object
                    pipes.append(Pipe(500))
            
            pipes = [pipe for pipe in pipes if pipe.x > -pipe.PIPE_TOP.get_width()]

            if bird.y + bird.img.get_height() >= base.y:
                game_active = False

        # Update the draw_window call to include show_debug
        draw_window(win, [bird], pipes, base, stats, game_started, game_active, generation)

    pygame.quit()
if __name__ == "__main__":
    main()


