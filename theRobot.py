import pychrono.core as chrono
import pychrono.irrlicht as chronoirr
import theBattleground

class theRobot():
    def __init__(self,system):

        self.mysystem = system
        self.createBody()
        self.createWheels()
        self.setWheelMaterial(0.9,1.4,0.0003,0.0003)
        self.enableContact()

    def __del__(self):
        self.mysystem.__del__()
        self.mbody1.__del__()
        self.Body_wheel_L.__del__()
        self.Body_wheel_R.__del__()
        self.mboxasset.__del__()

        self.Shape_Wheel_L.__del__()
        self.motor_L.__del__()
        self.motorpos_L.__del__()

        self.Shape_Wheel_R.__del__()
        self.motor_R.__del__()
        self.motorpos_R.__del__()

        self.wheelMaterial.__del__()

    def resetPos(self):
        self.mbody1.SetPos( chrono.ChVectorD(0,0,-0.2))
        self.mbody1.SetRot(chrono.ChQuaternionD(0,0,0,1))
        self.mbody1.SetNoSpeedNoAcceleration()


        self.Body_wheel_L.SetPos(chrono.ChVectorD(0,-0.5,0))
        self.Body_wheel_L.SetRot(chrono.ChQuaternionD(0.707,-0.707,0,0))
        self.Body_wheel_L.SetNoSpeedNoAcceleration()

        self.Body_wheel_R.SetPos(chrono.ChVectorD(0,-0.5,-0.3))
        self.Body_wheel_R.SetRot(chrono.ChQuaternionD(0.707,-0.707,0,0))
        self.Body_wheel_R.SetNoSpeedNoAcceleration()


    def createBody(self):
        # Create a fixed rigid body

        self.mbody1 = chrono.ChBody()
        self.mbody1.SetBodyFixed(False)
        self.mbody1.SetIdentifier(1)
        self.mbody1.SetPos( chrono.ChVectorD(0,0,-0.2))
        self.mbody1.SetRot(chrono.ChQuaternionD(0,0,0,1))
        self.mysystem.Add(self.mbody1)

        self.mboxasset = chrono.ChBoxShape()
        self.mboxasset.GetBoxGeometry().Size = chrono.ChVectorD(0.2,0.5,0.1)
        self.mbody1.AddAsset(self.mboxasset)

    def createWheels(self):
        # Left Wheel

        self.Body_wheel_L = chrono.ChBody()
        self.Body_wheel_L.SetBodyFixed(False)
        self.Body_wheel_L.SetPos(chrono.ChVectorD(0,-0.5,0))
        self.Body_wheel_L.SetRot(chrono.ChQuaternionD(0.707,-0.707,0,0))
        self.mysystem.Add(self.Body_wheel_L)

        self.Shape_Wheel_L = chrono.ChCylinderShape()
        self.Shape_Wheel_L.GetCylinderGeometry().radius = 0.03
        self.Shape_Wheel_L.GetCylinderGeometry().p1 = chrono.ChVectorD(0, 0.1, 0)
        self.Shape_Wheel_L.GetCylinderGeometry().p2 = chrono.ChVectorD(0, -0.01, 0)
        self.Body_wheel_L.AddAsset(self.Shape_Wheel_L)

        self.motor_L = chrono.ChLinkMotorRotationSpeed()
        self.motorpos_L = chrono.ChFrameD(chrono.ChVectorD(0,-0.5,0))
        self.motor_L.Initialize(self.mbody1,self.Body_wheel_L,self.motorpos_L)
        self.mysystem.Add(self.motor_L)

        # Right Wheel

        self.Body_wheel_R = chrono.ChBody()
        self.Body_wheel_R.SetBodyFixed(False)
        self.Body_wheel_R.SetPos(chrono.ChVectorD(0,-0.5,-0.3))
        self.Body_wheel_R.SetRot(chrono.ChQuaternionD(0.707,-0.707,0,0))
        self.mysystem.Add(self.Body_wheel_R)

        self.Shape_Wheel_R = chrono.ChCylinderShape()
        self.Shape_Wheel_R .GetCylinderGeometry().radius = 0.03
        self.Shape_Wheel_R .GetCylinderGeometry().p1 = chrono.ChVectorD(0, 0.1, 0)
        self.Shape_Wheel_R .GetCylinderGeometry().p2 = chrono.ChVectorD(0, -0.01, 0)
        self.Body_wheel_R.AddAsset(self.Shape_Wheel_R )

        self.motor_R = chrono.ChLinkMotorRotationSpeed()
        self.motorpos_R = chrono.ChFrameD(chrono.ChVectorD(0,-0.5,0))
        self.motor_R.Initialize(self.mbody1,self.Body_wheel_R,self.motorpos_R)
        self.mysystem.Add(self.motor_R)


    def setWheelMaterial(self, friction, damping, compliance, complianceT):
        self.wheelMaterial = chrono.ChMaterialSurfaceNSC()
        self.wheelMaterial.SetFriction(friction)
        self.wheelMaterial.SetDampingF(damping)
        self.wheelMaterial.SetCompliance (compliance)
        self.wheelMaterial.SetComplianceT(complianceT)

    def enableContact(self):
        self.mbody1.GetCollisionModel().ClearModel()
        self.mbody1.GetCollisionModel().AddBox(0.2,0.5,0.1) # hemi sizes
        self.mbody1.GetCollisionModel().BuildModel()
        self.mbody1.SetCollide(True)
        self.mbody1.SetMass(20)

        self.Body_wheel_L.GetCollisionModel().ClearModel()
        self.Body_wheel_L.GetCollisionModel().AddCylinder(0.2,0.5,0.1) # hemi sizes
        self.Body_wheel_L.GetCollisionModel().BuildModel()
        self.Body_wheel_L.SetCollide(True)
        self.Body_wheel_L.SetMaterialSurface(self.wheelMaterial)
        self.Body_wheel_L.SetMass(1)
        self.Body_wheel_L.AddAsset(chrono.ChTexture("Wheel_texture.png"))

        self.Body_wheel_R.GetCollisionModel().ClearModel()
        self.Body_wheel_R.GetCollisionModel().AddCylinder(0.2,0.5,0.1) # hemi sizes
        self.Body_wheel_R.GetCollisionModel().BuildModel()
        self.Body_wheel_R.SetCollide(True)
        self.Body_wheel_R.SetMaterialSurface(self.wheelMaterial)
        self.Body_wheel_R.SetMass(1)
        self.Body_wheel_R.AddAsset(chrono.ChTexture("Wheel_texture.png"))
