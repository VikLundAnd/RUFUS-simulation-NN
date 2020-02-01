import pychrono.core as chrono
import pychrono.irrlicht as chronoirr
import theBattleground
import theRobot
import time
from DQN import DQNAgent
import pygame
from random import randint
import random
from DQN import DQNAgent
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class Simulation():
    def __init__(self):

        self.amountOfSimulations = 0
        self.maxSpeed = 4
        self.score = 0
        self.previousScore = 0
        self.highscore = 0
        self.highscoreTime = 0
        #self.delayCounter = 0
        #self.delay = randint(5,10)
        self.keepRunning = True
        self.agent = DQNAgent()

        self.mysystem      = chrono.ChSystemNSC()
        self.ground = theBattleground.theBattleground(self.mysystem)
        self.createRobot(self.mysystem)
        self.createApplication()
        self.run()

    def createApplication(self):
        #  Create an Irrlicht application to visualize the system

        self.myapplication = chronoirr.ChIrrApp(self.mysystem, 'PyChrono example', chronoirr.dimension2du(1024,768))

        self.myapplication.AddTypicalSky()
        self.myapplication.AddTypicalLogo()
        self.myapplication.AddTypicalCamera(chronoirr.vector3df(0.6,0.6,0.8))
        self.myapplication.AddLightWithShadow(chronoirr.vector3df(2,4,2),    # point
                                         chronoirr.vector3df(0,0,0),    # aimpoint
                                         9,                 # radius (power)
                                         1,9,               # near, far
                                         30)                # angle of FOV

                 # ==IMPORTANT!== Use this function for adding a ChIrrNodeAsset to all items
                             # in the system. These ChIrrNodeAsset assets are 'proxies' to the Irrlicht meshes.
                             # If you need a finer control on which item really needs a visualization proxy in
                             # Irrlicht, just use application.AssetBind(myitem); on a per-item basis.

        self.myapplication.AssetBindAll();

                             # ==IMPORTANT!== Use this function for 'converting' into Irrlicht meshes the assets
                             # that you added to the bodies into 3D shapes, they can be visualized by Irrlicht!

        self.myapplication.AssetUpdateAll();
        self.myapplication.AddShadowAll();
        self.myapplication.SetShowInfos(True)

    def displayScore(self):
        print("Score: " + str(self.score) + "     Highscore: " + str(self.highscore))

    def checkIfDead(self):
        if (((self.robot.mbody1.GetRot()).Q_to_Rotv()).z < 2.5):
            print("tilt forward - DEAD")
            self.keepRunning = False

        if (((self.robot.mbody1.GetRot()).Q_to_Rotv()).z > 4):
            print("tilt backward - DEAD")
            self.keepRunning = False

    def doMove(self, prediction):

        self.checkIfDead()

        print("pred is: ")
        print(prediction)
        print(" ")

        if prediction[0][0] > prediction[0][1]:
            speed = (prediction[0][0] * self.maxSpeed)
            self.robot.motor_R.SetMotorFunction(chrono.ChFunction_Const(speed))
            self.robot.motor_L.SetMotorFunction(chrono.ChFunction_Const(speed))

        elif  prediction[0][1] > prediction[0][0]:
            speed = -(prediction[0][1] * self.maxSpeed)
            self.robot.motor_R.SetMotorFunction(chrono.ChFunction_Const(speed))
            self.robot.motor_L.SetMotorFunction(chrono.ChFunction_Const(speed))

        else:
            self.robot.motor_R.SetMotorFunction(chrono.ChFunction_Const(0))
            self.robot.motor_L.SetMotorFunction(chrono.ChFunction_Const(0))


    def createRobot(self, system):
        try:
            del self.robot
        except:
            pass
        self.robot = theRobot.theRobot(system)

    def restart(self):
        print("restart called")

        del self.myapplication
        del self.mysystem
        self.mysystem      = chrono.ChSystemNSC()
        self.ground = theBattleground.theBattleground(self.mysystem)
        self.createRobot(self.mysystem)
        self.createApplication()

        self.mysystem.SetChTime(0)
        self.keepRunning = True

        if self.score > self.highscore:
            self.highscore = self.score
            self.highscoreTime = self.amountOfSimulations

        self.previousScore = self.score
        self.score = 0

    def run(self):
        self.myapplication.SetTimestep(0.1)
        self.myapplication.SetTryRealtime(False)

        while self.amountOfSimulations <= 500:
            self.restart()

            while(self.myapplication.GetDevice().run() and self.keepRunning):
                self.agent.epsilon = 80 - self.amountOfSimulations

                #get old state
                state_old = self.agent.get_state(self.robot)

                self.myapplication.BeginScene()
                self.myapplication.DrawAll()
                self.myapplication.DoStep()
                self.myapplication.EndScene()
                self.checkIfDead()


                #perform random actions based on agent.epsilon, or choose the action
                if randint(0, 200) < self.agent.epsilon:
                    #prediction = to_categorical(random.random(), num_classes=2)
                    prediction = [[random.random(),random.random()]]
                    print("random action")
                else:
                    # predict action based on the old state
                    print("AI action")
                    prediction = self.agent.model.predict(state_old.reshape((1,2)))
                    #final_move = to_categorical(np.argmax(prediction[0]), num_classes=2)

                #perform new move and get new state
                self.doMove(prediction)
                state_new = self.agent.get_state(self.robot)

                #self.delay = randint(5, 10)
                self.delayCounter = 0


                self.displayScore()

                self.score += 1
                #self.delayCounter += 1

            #set reward for the new state
            reward = self.agent.set_reward(self.score, self.highscore, self.previousScore, self.amountOfSimulations, self.highscoreTime)

            #train short memory base on the new action and state
            self.agent.train_short_memory(state_old, prediction, reward, state_new, self.keepRunning)

            # store the new data into a long term memory
            self.agent.remember(state_old, prediction, reward, state_new, self.keepRunning)


            self.amountOfSimulations += 1
            self.agent.replay_new(self.agent.memory)

        self.agent.model.save_weights('weights.hdf5')

Simulation()
