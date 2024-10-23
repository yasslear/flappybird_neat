import pygame
import os
import random

WIN_WIDTH = 500
WIN_HEIGHT = 800

BIRD_IMGS = [pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird1.png"))),
             pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird2.png"))),
             pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird3.png")))]
PIPE_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "pipe.png")))
BASE_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "base.png")))
BG_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bg.png")))

class Bird:
    IMGS = BIRD_IMGS
    MAX_ROTATION = 25
    ROT_VEL = 20
    ANIMATION_TIME = 5

    def __init__(self, x, y):
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
        d = self.vel * self.tick_count + 1.5 * self.tick_count ** 2

        if d >= 16:
            d = 16

        if d < 0:
            d -= 2

        self.y += d

        if d < 0 or self.y < self.height + 50:
            if self.tilt < self.MAX_ROTATION:
                self.tilt = self.MAX_ROTATION
        else:
            if self.tilt > -90:
                self.tilt -= self.ROT_VEL
        if self.y > 760:
            self.y = 760

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


def draw_window(win, bird, pipes, base, score, game_started, game_active):
    win.blit(BG_IMG, (0, 0))  

    for pipe in pipes:
        pipe.draw(win)

    base.draw(win)
    bird.draw(win)

    
    font = pygame.font.SysFont("comicsans", 25)

    
    score_label = font.render(f"Score: {score}", 1, (255, 255, 255))  # White color
    win.blit(score_label, (10, 10))

    
    if not game_started:
        
        start_label = font.render("Left click or press space bar to play", 1, (255, 255, 255))
        win.blit(start_label, (WIN_WIDTH / 2 - start_label.get_width() / 2, WIN_HEIGHT / 3 - 50))

        
        ai_label = font.render("Press A to let NEAT AI play", 1, (255, 255, 255))
        win.blit(ai_label, (WIN_WIDTH / 2 - ai_label.get_width() / 2, WIN_HEIGHT / 3 + 10))

    if not game_active and game_started:
        game_over_label = font.render("Game Over! Press R to restart", 1, (255, 0, 0))  # Red color
        win.blit(game_over_label, (WIN_WIDTH / 2 - game_over_label.get_width() / 2, WIN_HEIGHT / 3 - 50))

    pygame.display.update()


def main():
    pygame.init()
    bird = Bird(230, 350)
    base = Base(730)
    pipes = [Pipe(500)]
    win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    clock = pygame.time.Clock()

    # Flags for game state
    game_started = False  # Game starts paused, waiting for first input
    game_active = True    # Game continues unless there's a collision
    score = 0             # Initial score is zero

    run = True

    while run:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            # Handle keypresses
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and not game_started:
                    game_started = True  # Start the game on first key press

                elif event.key == pygame.K_SPACE and game_active:
                    bird.jump()  # Jump if the game is active

                elif event.key == pygame.K_a and not game_started:
                    # NEAT AI functionality placeholder
                    print("AI learning mode triggered (to be implemented)")

                elif event.key == pygame.K_r and not game_active:
                    # Reset the game state
                    bird = Bird(230, 350)
                    pipes = [Pipe(500)]
                    score = 0
                    game_started = False  # Reset to the initial state
                    game_active = True    # Allow restarting the game

            # Handle mouse clicks
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1 and not game_started:
                    game_started = True  # Start the game on first mouse click
                elif event.button == 1 and game_active:
                    bird.jump()  # Jump if the game is active

        # Only move everything if the game has started and is active
        if game_started and game_active:
            for pipe in pipes:
                pipe.move()

            bird.move()
            base.move()

            # Check for collisions with pipes or the base
            for pipe in pipes:
                if pipe.collide(bird):
                    game_active = False  # Stop everything on collision
                    break

            if bird.y + bird.img.get_height() >= base.y:  # Collision with base
                game_active = False

        draw_window(win, bird, pipes, base, score, game_started, game_active)  # Always draw the window

    pygame.quit()

if __name__ == "__main__":
    main()

