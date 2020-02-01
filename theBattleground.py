import theRobot
import pychrono.core as chrono
import pychrono.irrlicht as chronoirr

class theBattleground():
    def __init__(self, system):

        self.mysystem = system
        self.createGround()
        #set surface material to BRICK MATERIAL
        self.setSurfaceMaterial(0.5,0.2,0.0000001,0.0000001)
        self.enableContact()

    def resetPos(self):
        self.createGround()
        #set surface material to BRICK MATERIAL


    def createGround(self):
        # Create ground

        self.ground = chrono.ChBody()
        self.mysystem.AddBody(self.ground)
        self.ground.SetIdentifier(-1)
        self.ground.SetPos(chrono.ChVectorD(0,-1,0))
        self.ground.SetName("ground")
        self.ground.SetBodyFixed(True)

        self.box_ground = chrono.ChBoxShape()
        self.box_ground.GetBoxGeometry().Size = chrono.ChVectorD(5,0.01,5)
        self.ground.AddAsset(self.box_ground)

    def setSurfaceMaterial(self, friction, damping, compliance, complianceT):
        self.surface = chrono.ChMaterialSurfaceNSC()
        self.surface.SetFriction(friction)
        self.surface.SetDampingF(damping)
        self.surface.SetCompliance (compliance)
        self.surface.SetComplianceT(complianceT)


    def enableContact(self):
        # Enable contact
        self.ground.GetCollisionModel().ClearModel()
        self.ground.GetCollisionModel().AddBox(5,0.01,5) # hemi sizes
        self.ground.GetCollisionModel().BuildModel()
        self.ground.SetMaterialSurface(self.surface)
        self.ground.SetCollide(True)
