import pygame
#from pygame.locals import *
import sys
import random

class DoodleJump:
    def __init__(self):
        self.height = 800
        self.width = 600
        self.screen = pygame.display.set_mode((self.width, self.height)) # setting screen resolution
        self.green = pygame.image.load("assets/green.png").convert_alpha() # loading green platform
        pygame.font.init()
        self.score = 0 # init score = 0
        self.font = pygame.font.SysFont("Arial", 25)
        self.blue = pygame.image.load("assets/blue.png").convert_alpha() # loading moving blue platform
        self.red = pygame.image.load("assets/red.png").convert_alpha() # breakable platform unbroken
        self.red_1 = pygame.image.load("assets/red_1.png").convert_alpha() # breakable platform broken
        self.playerRight = pygame.image.load("assets/right.png").convert_alpha() # doodler faing right
        self.playerRight_1 = pygame.image.load("assets/right_1.png").convert_alpha() # doodler facing right jumping
        self.playerLeft = pygame.image.load("assets/left.png").convert_alpha() # doodler facing left
        self.playerLeft_1 = pygame.image.load("assets/left_1.png").convert_alpha() # doodler facing left jumping
        self.spring = pygame.image.load("assets/spring.png").convert_alpha() # spring wound
        self.spring_1 = pygame.image.load("assets/spring_1.png").convert_alpha() # spring unwound
        self.direction = 0

        # initial position of player
        self.playerx = self.width // 2
        self.playery = self.height//2

        self.platforms = [[self.width // 2, self.height+100, 0, 0]]
        self.springs = []
        self.cameray = 0 # distance the top of the screen has travelled from beginning.
        self.jump = 0
        self.gravity = 0
        self.xmovement = 0
    
    def updatePlayer(self):
        if not self.jump:        
            self.playery += self.gravity
            self.gravity += 1
        elif self.jump:
            self.playery -= self.jump
            self.jump -= 1
        key = pygame.key.get_pressed()
        if key[pygame.K_RIGHT]:
            if self.xmovement < 10:
                self.xmovement += 1
            self.direction = 0

        elif key[pygame.K_LEFT]:
            if self.xmovement > -10:
                self.xmovement -= 1
            self.direction = 1
        else:
            if self.xmovement > 0:
                self.xmovement -= 1
            elif self.xmovement < 0:
                self.xmovement += 1
        if self.playerx > 850:
            self.playerx = -50
        elif self.playerx < -50:
            self.playerx = 850
        self.playerx += self.xmovement
        if self.playery - self.cameray <= 200:
            self.cameray -= 10
        if not self.direction:
            if self.jump:
                self.screen.blit(self.playerRight_1, (self.playerx, self.playery - self.cameray))
            else:
                self.screen.blit(self.playerRight, (self.playerx, self.playery - self.cameray))
        else:
            if self.jump:
                self.screen.blit(self.playerLeft_1, (self.playerx, self.playery - self.cameray))
            else:
                self.screen.blit(self.playerLeft, (self.playerx, self.playery - self.cameray))

    def updatePlatforms(self):
        for p in self.platforms:
            rect = pygame.Rect(p[0], p[1], self.green.get_width() - 10, self.green.get_height())
            player = pygame.Rect(self.playerx, self.playery, self.playerRight.get_width() - 10, self.playerRight.get_height())
            if rect.colliderect(player) and self.gravity and self.playery < (p[1] - self.cameray):
                if p[2] != 2:
                    self.jump = 15
                    self.gravity = 0
                else:
                    p[-1] = 1
            if p[2] == 1:
                if p[-1] == 1:
                    p[0] += 5
                    if p[0] > 550:
                        p[-1] = 0
                else:
                    p[0] -= 5
                    if p[0] <= 0:
                        p[-1] = 1

    def drawPlatforms(self):
        check = self.platforms[1][1] - self.cameray
        if check > self.height:  # check if second last platform in window has crossed height. then, have to draw new platform above
            # add a platform
            platform = self.get_platform()
            x = random.randint(0, self.width - 100)
            on = self.platforms[-1][1] - 50 # change this 50 number TODO
            self.platforms.append([x, on, platform, 0])
            
            # adding a spring to the platform with 1/10 * P(platform being 0) probability.
            coords = self.platforms[-1]
            check = random.randint(0, 1000)
            if check > 900 and platform == 0:
                self.springs.append([coords[0], coords[1] - 25, 0]) # add spring
            
            # remove first platform since it is out of screen, update score.
            self.platforms.pop(0)
            self.score += 100
        
        # assign platform images
        for p in self.platforms:
            if p[2] == 0:
                self.screen.blit(self.green, (p[0], p[1] - self.cameray))
            elif p[2] == 1:
                self.screen.blit(self.blue, (p[0], p[1] - self.cameray))
            elif p[2] == 2:
                if not p[3]:
                    self.screen.blit(self.red, (p[0], p[1] - self.cameray))
                else:
                    self.screen.blit(self.red_1, (p[0], p[1] - self.cameray))
        
        # add spring image
        for spring in self.springs:
            if spring[-1]:
                self.screen.blit(self.spring_1, (spring[0], spring[1] - self.cameray))
            else:
                self.screen.blit(self.spring, (spring[0], spring[1] - self.cameray))
            
            # add jump configuration
            if pygame.Rect(spring[0], spring[1], self.spring.get_width(), self.spring.get_height()).colliderect(pygame.Rect(self.playerx, self.playery, self.playerRight.get_width(), self.playerRight.get_height())):
                self.jump = 30
                self.cameray -= 40

    def generatePlatforms(self):
        on = self.height # height of window

        # only for the initial frame of the game, platforms are drawn at a height of 50px each,
        # from -100px to self.height.
        while on > -100:
            x = random.randint(0, self.width - 100) # width of window
            platform = self.get_platform()
            self.platforms.append([x, on, platform, 0])
            on -= 50

    def drawGrid(self):
        """
        Method to draw background grid on the screen.
        """
        for x in range(80):
            pygame.draw.line(self.screen, (222,222,222), (x * 12, 0), (x * 12, self.height))
            pygame.draw.line(self.screen, (222,222,222), (0, x * 12), (self.width, x * 12))
    
    def get_platform(self):
        """
        Returns type of platform with probabilities - G=8/10, blue and red=1/10 each
        """
        platform = random.randint(0, 1000)
        if platform < 800:
            platform = 0
        elif platform < 900:
            platform = 1
        else:
            platform = 2
        return platform

    def run(self):
        clock = pygame.time.Clock()
        self.generatePlatforms()

        while True:
            print("cameray = ", self.cameray)
            print("playery = ", self.playery)
            self.screen.fill((255,255,255))
            clock.tick(60)
            
            # quitting game
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
            
            # restarting game
            if self.playery - self.cameray > self.height + 100: # top of screen - pos of doodler > 700 => doodler out of window.
                self.cameray = 0
                self.score = 0
                self.springs = []
                self.platforms = [[self.width//2, self.height+100, 0, 0]]
                self.generatePlatforms()
                self.playerx = self.width//2
                self.playery = self.height//2
            
            self.drawGrid()
            self.drawPlatforms()
            self.updatePlayer()
            self.updatePlatforms()

            # display score
            self.screen.blit(self.font.render(str(self.score), -1, (0, 0, 0)), (25, 25))
            
            pygame.display.flip() 


DoodleJump().run()
