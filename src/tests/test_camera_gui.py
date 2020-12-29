from src.visualisers.camera_gui import (
    BasicCameraGUI, PolarGUI, QuadPolar, CamGUIReduced, DeComGUI, PBTDeComGUI, PolarPBT, PBTPolarQuad
)

def test_basic():
    basic = BasicCameraGUI()
    basic.run()

def test_polar():
    polar = PolarGUI()
    polar.run()

def test_quad():
    qp = QuadPolar()
    qp.run()

def test_reduced():
    reduced = CamGUIReduced()
    reduced.run()

def test_decom():
    decom = DeComGUI()
    decom.run()

def test_pbt_decom():
    decom = PBTDeComGUI()
    decom.run()

def test_polar_pbt():
    ppb_decom = PolarPBT(stages=4)
    ppb_decom.run()

def test_quad_decom():
    polar_quad = PBTPolarQuad()
    polar_quad.run()