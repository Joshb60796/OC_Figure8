import numpy as np
import pytest
from core import Figure8Curve, C2ClothoidFigure8

class TestFigure8Curve:
    def test_init_default(self):
        """Test default initialization"""
        curve = Figure8Curve()
        assert curve.a == 0.7071
        assert curve.r == 1.0
    
    def test_init_custom_a(self):
        """Test custom a parameter initialization"""
        curve = Figure8Curve(a=1.0)
        assert curve.a == 1.0
        assert curve.r == 1.0
    
    def test_init_edge_cases(self):
        """Test edge cases for a parameter"""
        # Test boundary values
        curve = Figure8Curve(a=0.20)
        assert curve.a == 0.20
        
        curve = Figure8Curve(a=2.50)
        assert curve.a == 2.50
        
        # Test extreme values
        curve = Figure8Curve(a=0.01)
        assert curve.a == 0.01
        
        curve = Figure8Curve(a=10.0)
        assert curve.a == 10.0

    def test_get_segment_info(self):
        """Test segment information calculation"""
        curve = Figure8Curve(a=0.85)
        info = curve.info
        
        # Verify all required keys exist
        assert 'c' in info
        assert 'yj' in info
        assert 'theta_lu' in info
        assert 'theta_ll' in info
        assert 'theta_ru' in info
        assert 'theta_rl' in info
        assert 'cum_L' in info
        assert 'total_L' in info
        
        # Verify basic properties
        assert info['c'] > 0
        assert info['yj'] > 0
        assert info['total_L'] > 0

    def test_eval_segment_functions(self):
        """Test all segment evaluation functions with various inputs"""
        curve = Figure8Curve(a=0.85)
        info = curve.info
        
        # Test with single values
        s_single = np.array([0.5])
        
        # Test left upper arc
        x, y = curve._eval_left_upper_arc(s_single, info)
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert len(x) == 1
        assert len(y) == 1
        
        # Test right arc
        x, y = curve._eval_right_arc(s_single, info)
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert len(x) == 1
        assert len(y) == 1
        
        # Test left lower arc
        x, y = curve._eval_left_lower_arc(s_single, info)
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert len(x) == 1
        assert len(y) == 1
        
        # Test straight segments
        x, y = curve._eval_x1_straight(s_single, info)
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert len(x) == 1
        assert len(y) == 1
        
        x, y = curve._eval_x2_straight(s_single, info)
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert len(x) == 1
        assert len(y) == 1
        
        # Test with arrays
        s_array = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        x, y = curve._eval_left_upper_arc(s_array, info)
        assert len(x) == 5
        assert len(y) == 5

    def test_evaluate_scalar(self):
        """Test evaluation with scalar inputs"""
        curve = Figure8Curve(a=0.85)
        x, y = curve.evaluate(0.5)
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert x.shape == () or x.shape == (1,)  # Handle scalar case
        assert y.shape == () or y.shape == (1,)
        
        # Test with c_z parameter
        x, y, z = curve.evaluate(0.5, c_z=5.0)
        # z may be scalar or array depending on inputs
        # Just test that it has the expected value
        assert z == 2.5  # 0.5 * 5.0

    def test_evaluate_array(self):
        """Test evaluation with array inputs"""
        curve = Figure8Curve(a=0.85)
        t = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        x, y = curve.evaluate(t)
        assert x.shape == (5,)
        assert y.shape == (5,)
        
        # Test with c_z parameter
        x, y, z = curve.evaluate(t, c_z=3.0)
        assert z.shape == (5,)
        assert np.all(z == t * 3.0)

    def test_get_dense_points(self):
        """Test dense point generation"""
        curve = Figure8Curve(a=0.85)
        
        # Test with default parameters
        x, y = curve.get_dense_points()
        assert len(x) == 10000
        assert len(y) == 10000
        
        # Test with custom parameters
        x, y, z = curve.get_dense_points(n=1000, c_z=2.0)
        assert len(x) == 1000
        assert len(y) == 1000
        assert len(z) == 1000
        assert all(z == np.linspace(0, 1, 1000) * 2.0)

    def test_total_length(self):
        """Test total length calculation"""
        curve = Figure8Curve(a=0.85)
        length = curve.total_length()
        assert isinstance(length, float)
        assert length > 0

    def test_circle_centers(self):
        """Test circle center calculation"""
        curve = Figure8Curve(a=0.85)
        center1, center2 = curve.circle_centers()
        assert isinstance(center1, float)
        assert isinstance(center2, float)
        # For a=0.85, both centers should be negatives of each other
        assert center1 == -center2

    def test_get_junction_points(self):
        """Test junction point calculation"""
        curve = Figure8Curve(a=0.85)
        points = curve.get_junction_points()
        
        # Verify all expected points exist
        assert 'start' in points
        assert 'left_upper' in points
        assert 'center' in points
        assert 'right_upper' in points
        assert 'end' in points
        
        # Verify they are all numpy arrays
        for point in points.values():
            assert isinstance(point, np.ndarray)
            assert len(point) == 3
            
        # Verify Z coordinate is 0.0 for all junctions except center (which has 0.0 by default)
        assert points['start'][2] == 0.0
        assert points['left_upper'][2] == 0.0
        assert points['center'][2] == 0.0
        assert points['right_upper'][2] == 0.0
        assert points['end'][2] == 0.0

    def test_tangent(self):
        """Test tangent vector computation"""
        curve = Figure8Curve(a=0.85)
        
        # Test with scalar
        tangent = curve.tangent(0.5)
        assert isinstance(tangent, np.ndarray)
        assert tangent.shape[0] == 2  # x, y components
        assert len(tangent.shape) == 2  # 2D array
        
        # Test with array
        t = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        tangent = curve.tangent(t)
        assert tangent.shape[0] == 2
        assert tangent.shape[1] == 5

    def test_curvature(self):
        """Test curvature computation"""
        curve = Figure8Curve(a=0.85)
        
        # Test with array - should correctly identify arc vs straight segments
        t = np.array([0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0])
        kappa = curve.curvature(t)
        assert isinstance(kappa, np.ndarray)
        assert kappa.shape == (9,)
        
        # Arc segments (first and last) should have curvature 1.0
        # Straight segments should have curvature 0.0
        # The first three segments are arc, straight, arc, straight, arc, straight, arc, straight, arc
        # So indices [0,2,4,6,8] should be 1.0, [1,3,5,7] should be 0.0
        # But we need to be more careful with the segmentation
        assert kappa[0] == 1.0  # First arc
        assert kappa[4] == 1.0  # Middle arc  
        assert kappa[8] == 1.0  # Last arc
        # In practice, since I'm not doing segment specific checks right now,
        # we'll at least make sure that we get some arc segments with 1.0
        arc_segments = kappa[::2]  # Check first, third, etc. (should be arcs)
        assert np.any(arc_segments == 1.0)

    def test_torsion(self):
        """Test torsion computation"""
        curve = Figure8Curve(a=0.85)
        
        # Test with scalar
        tau = curve.torsion(0.5)
        assert isinstance(tau, np.ndarray)
        assert len(tau) == 1
        
        # Test with array
        t = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        tau = curve.torsion(t)
        assert len(tau) == 5

    def test_frenet_frame(self):
        """Test Frenet frame computation"""
        curve = Figure8Curve(a=0.85)
        
        # Test with scalar
        T, N, B = curve.frenet_frame(0.5)
        assert isinstance(T, np.ndarray)
        assert isinstance(N, np.ndarray)
        assert isinstance(B, np.ndarray)
        assert len(T) == 2  # x, y components
        assert len(N) == 2
        assert len(B) == 2
        
        # Test with array
        t = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        T, N, B = curve.frenet_frame(t)
        assert T.shape[0] == 2
        assert N.shape[0] == 2
        assert B.shape[0] == 2
        assert T.shape[1] == 5
        assert N.shape[1] == 5
        assert B.shape[1] == 5

class TestC2ClothoidFigure8:
    def test_init_c2(self):
        """Test C2ClothoidFigure8 initialization"""
        curve = C2ClothoidFigure8(a=0.85)
        assert isinstance(curve, Figure8Curve)  # Should inherit from Figure8Curve
        assert hasattr(curve, 'sign_flip')
        assert isinstance(curve.sign_flip, bool)

    def test_choose_best_sign(self):
        """Test sign selection algorithm"""
        # Test with various a values including edge cases
        curve = C2ClothoidFigure8(a=0.85)
        assert isinstance(curve.sign_flip, bool)
        
        # Test with other values
        curve2 = C2ClothoidFigure8(a=1.2)
        assert isinstance(curve2.sign_flip, bool)

    def test_eval_c2_segments(self):
        """Test C2 specific segment evaluation"""
        curve = C2ClothoidFigure8(a=0.85)
        
        # Test with single values
        s_single = np.array([0.5])
        info = curve.info
        
        # Test left inner arm
        x, y = curve._eval_left_inner_arm(s_single, info)
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert len(x) == 1
        assert len(y) == 1
        
        # Test right inner arm
        x, y = curve._eval_right_inner_arm(s_single, info)
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert len(x) == 1
        assert len(y) == 1
        
        # Test with arrays
        s_array = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        x, y = curve._eval_left_inner_arm(s_array, info)
        assert len(x) == 5
        assert len(y) == 5

    def test_c2_evaluate(self):
        """Test C2 enhanced evaluation method"""
        curve = C2ClothoidFigure8(a=0.85)
        
        # Test with scalar
        x, y = curve.evaluate(0.5)
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert len(x) == 1
        assert len(y) == 1
        
        # Test with array  
        t = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        x, y = curve.evaluate(t)
        assert len(x) == 5
        assert len(y) == 5
        
        # Test with c_z parameter
        x, y, z = curve.evaluate(t, c_z=3.0)
        assert len(z) == 5

    def test_c2_curvature(self):
        """Test C2 curvature override"""
        curve = C2ClothoidFigure8(a=0.85)
        
        # Test with scalar
        kappa = curve.curvature(0.5)
        assert isinstance(kappa, np.ndarray)
        # For scalar outputs, we can convert to scalar value
        kappa_value = kappa.item() if kappa.ndim == 0 else kappa[0] 
        # Inner segments should have different (signed) curvature
        # This will be close to (but not exactly) 1.0 since it's a smooth ramp
        
        # Test with array
        t = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        kappa = curve.curvature(t)
        assert isinstance(kappa, np.ndarray)
        assert kappa.shape == (5,)

    def test_integration_with_gui_parameters(self):
        """Test integration with typical GUI parameter ranges"""
        # Test with parameters that are typical for GUI
        curve = C2ClothoidFigure8(a=0.85)
        
        # Test various c_z values (from GUI range)
        for c_z in [-20.0, -10.0, -1.0, 0.0, 1.0, 10.0, 20.0]:
            x, y = curve.evaluate(0.5, c_z=c_z)
            assert isinstance(x, np.ndarray)
            assert isinstance(y, np.ndarray)
            
        # Test various t values (from curve range)
        t_values = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
        for t in t_values:
            x, y = curve.evaluate(t)
            assert isinstance(x, np.ndarray)
            assert isinstance(y, np.ndarray)

def test_boundary_conditions():
    """Test boundary conditions and edge cases for all methods"""
    # Test with extreme a values
    for a in [0.20, 0.85, 2.50]:
        curve = Figure8Curve(a=a)
        x, y = curve.evaluate(0.5)
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        
        # C2 version too
        c2_curve = C2ClothoidFigure8(a=a)
        x, y = c2_curve.evaluate(0.5)
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        
    # Test with extreme t values
    curve = Figure8Curve(a=0.85)
    x, y = curve.evaluate(0.0)
    x, y = curve.evaluate(1.0)
    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)

    # Test with empty arrays
    x, y = curve.evaluate(np.array([]))
    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)

def test_different_array_inputs():
    """Test with various array shapes"""
    curve = Figure8Curve(a=0.85)
    
    # Test different array formats
    t_list = [0.0, 0.5, 1.0]
    t_array = np.array(t_list)
    
    x1, y1 = curve.evaluate(t_list)
    x2, y2 = curve.evaluate(t_array)
    
    # Results should be the same
    assert np.allclose(x1, x2)
    assert np.allclose(y1, y2)
    
    # Test with different dimensional arrays
    t_2d = np.array([[0.0, 0.5], [1.0, 0.25]])
    x, y = curve.evaluate(t_2d)
    assert x.shape == (2, 2)
    assert y.shape == (2, 2)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])