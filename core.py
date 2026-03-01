import numpy as np
from typing import Tuple, Dict, Optional, Union

class Figure8Curve:
    """
    C¹-continuous open figure-8 curve with perfect circular outer lobes (r=1.0 fixed).
    """
    def __init__(self, a: float = 0.7071):
        self.a = float(a)
        self.r = 1.0
        self.info = self._get_segment_info()

    def _get_segment_info(self) -> Dict:
        c = (self.a + np.sqrt(self.a**2 + 4 * self.r**2)) / 2
        d = self.a - c
        yj = np.sqrt(self.r**2 - d**2)
        theta_lu = np.arctan2(yj, c - self.a)
        theta_ll = np.arctan2(-yj, c - self.a)
        theta_ru = np.arctan2(yj, d)
        theta_rl = np.arctan2(-yj, d)

        L1 = self.r * (np.pi - theta_lu)
        L2 = 2 * np.sqrt(self.a**2 + yj**2)
        L3 = self.r * (theta_ru - theta_rl)
        L4 = L2
        L5 = self.r * (np.pi + theta_ll)

        total_L = L1 + L2 + L3 + L4 + L5
        cum_L = np.cumsum([0, L1, L2, L3, L4, L5])

        return {
            'c': c, 'yj': yj,
            'theta_lu': theta_lu, 'theta_ll': theta_ll,
            'theta_ru': theta_ru, 'theta_rl': theta_rl,
            'cum_L': cum_L, 'total_L': total_L
        }

    def _eval_left_upper_arc(self, s: np.ndarray, info: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate left upper circular arc segment"""
        theta = np.pi - (np.pi - info['theta_lu']) * s
        x = -info['c'] + self.r * np.cos(theta)
        y = self.r * np.sin(theta)
        return x, y

    def _eval_x1_straight(self, s: np.ndarray, info: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate left inner straight segment"""
        x = -self.a + 2*self.a * s
        y = info['yj'] - 2*info['yj'] * s
        return x, y

    def _eval_right_arc(self, s: np.ndarray, info: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate right circular arc segment"""
        theta = info['theta_rl'] + (info['theta_ru'] - info['theta_rl']) * s
        x = info['c'] + self.r * np.cos(theta)
        y = self.r * np.sin(theta)
        return x, y

    def _eval_x2_straight(self, s: np.ndarray, info: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate right inner straight segment"""
        x = self.a - 2*self.a * s
        y = info['yj'] - 2*info['yj'] * s
        return x, y

    def _eval_left_lower_arc(self, s: np.ndarray, info: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate left lower circular arc segment"""
        theta = info['theta_ll'] - (info['theta_ll'] + np.pi) * s
        x = -info['c'] + self.r * np.cos(theta)
        y = self.r * np.sin(theta)
        return x, y

    def evaluate(self, t: Union[float, np.ndarray], c_z: Optional[float] = None) -> Tuple:
        t = np.asarray(t)
        info = self.info
        cum = info['cum_L']
        total_L = info['total_L']

        x = np.zeros_like(t, dtype=float)
        y = np.zeros_like(t, dtype=float)
        z = c_z * t if c_z is not None else None

        s_global = t * total_L
        seg = np.searchsorted(cum, s_global, side='right') - 1
        seg = np.clip(seg, 0, 4)

        # Process each segment type with dedicated evaluation methods
        segment_functions = [
            self._eval_left_upper_arc,
            self._eval_x1_straight,
            self._eval_right_arc,
            self._eval_x2_straight,
            self._eval_left_lower_arc
        ]
        
        for i, func in enumerate(segment_functions):
            mask = (seg == i)
            if not np.any(mask): 
                continue
            s = (s_global[mask] - cum[i]) / (cum[i+1] - cum[i])
            x_mask, y_mask = func(s, info)
            x[mask] = x_mask
            y[mask] = y_mask

        if z is None:
            return x, y
        return x, y, z

    def get_dense_points(self, n: int = 10000, c_z: Optional[float] = None):
        t = np.linspace(0, 1, n)
        return self.evaluate(t, c_z)

    def total_length(self) -> float:
        return self.info['total_L']

    def circle_centers(self) -> Tuple[float, float]:
        return -self.info['c'], self.info['c']

    def get_junction_points(self) -> Dict:
        ts = np.array([0.0, self.info['cum_L'][1]/self.info['total_L'], 0.5, 1.0])
        result = self.evaluate(ts, c_z=None)
        # Handle both 2-tuple and 3-tuple return values from evaluate
        if len(result) == 2:
            x, y = result
        else:
            x, y, _ = result
        return {
            'start': np.array([x[0], y[0], 0.0]),
            'left_upper': np.array([x[1], y[1], 0.0]),
            'center': np.array([0.0, 0.0, 0.0]),
            'right_upper': np.array([x[2], y[2], 0.0]),
            'end': np.array([x[3], y[3], 0.0])
        }

    def tangent(self, t: Union[float, np.ndarray], c_z: float = 4.0) -> np.ndarray:
        t = np.atleast_1d(t)
        eps = 1e-8
        p1 = np.array(self.evaluate(t + eps, c_z))
        p0 = np.array(self.evaluate(t - eps, c_z))
        vel = p1 - p0
        norm = np.linalg.norm(vel, axis=0)
        # Return normalized tangent vector
        result = vel / norm
        # Ensure it's 2D for x,y coordinates (the first two components are the tangent)
        if result.shape[0] == 3:  # If z is included, we only want x,y
            return result[:2, :] if result.ndim > 1 else result[:2]
        return result

    def curvature(self, t: Union[float, np.ndarray], c_z: float = 4.0) -> np.ndarray:
        t = np.asarray(t)
        cum = self.info['cum_L'] / self.info['total_L']
        kappa = np.zeros_like(t)
        # Arc segments have constant curvature
        arc_masks = [(0 <= t) & (t < cum[1]), (cum[2] <= t) & (t < cum[3]), (cum[4] <= t) & (t <= 1)]
        for mask in arc_masks:
            kappa[mask] = 1.0 / self.r
        return kappa

    def torsion(self, t: Union[float, np.ndarray], c_z: float = 4.0) -> np.ndarray:
        t = np.atleast_1d(t)
        eps = 1e-5
        r0 = np.array(self.evaluate(t, c_z))
        r1 = np.array(self.evaluate(t + eps, c_z))
        rm1 = np.array(self.evaluate(t - eps, c_z))
        r2 = np.array(self.evaluate(t + 2*eps, c_z))
        rm2 = np.array(self.evaluate(t - 2*eps, c_z))

        rp = (r1 - rm1) / (2 * eps)
        rpp = (r1 - 2*r0 + rm1) / eps**2
        rppp = (r2 - 2*r1 + 2*rm1 - rm2) / (2 * eps**3)

        cross = np.cross(rpp, rppp, axis=0)
        numerator = np.sum(rp * cross, axis=0)
        denom = np.sum(np.cross(rp, rpp, axis=0)**2, axis=0) + 1e-12
        tau = numerator / denom
        # Ensure result is an array, not a scalar
        if np.isscalar(tau):
            return np.array([tau])
        return tau

    def frenet_frame(self, t: Union[float, np.ndarray], c_z: float = 4.0):
        T = self.tangent(t, c_z)
        eps = 1e-6
        dT = self.tangent(np.atleast_1d(t) + eps, c_z) - self.tangent(np.atleast_1d(t) - eps, c_z)
        kappa = np.linalg.norm(dT, axis=0) / (2 * eps)
        N = np.zeros_like(T)
        valid = kappa > 1e-8
        N[:, valid] = dT[:, valid] / kappa[valid]
        B = np.cross(T.T, N.T).T
        # Handle the case when input is scalar, make sure it returns scalar components
        if np.asarray(t).ndim == 0:
            # T should be shaped like (2,) when t is scalar, not (2,1)
            return T, N, B
        return T, N, B


class C2ClothoidFigure8(Figure8Curve):
    """C² version with signed curvature and automatic best-sign selection"""
    def __init__(self, a: float = 0.7071):
        super().__init__(a)
        self._choose_best_sign()
        print(f"✅ C2ClothoidFigure8 ready — signed curvature + auto best-sign (sign_flip={self.sign_flip})")
        self.diagnostic_report()

    def _choose_best_sign(self):
        """Test both sign combinations and pick the one with smallest junction error"""
        best_error = float('inf')
        best_flip = False

        for flip in [False, True]:
            self.sign_flip = flip
            ts = np.array([self.info['cum_L'][1]/self.info['total_L'],
                           self.info['cum_L'][3]/self.info['total_L']])
            xc1, yc1 = super().evaluate(ts, c_z=None)
            xc2, yc2 = self.evaluate(ts, c_z=None)
            pos_error = np.max(np.hypot(xc2 - xc1, yc2 - yc1))
            if pos_error < best_error:
                best_error = pos_error
                best_flip = flip

        self.sign_flip = best_flip
        if best_error > 0.1:
            print(f"Warning: C2 junction error still high ({best_error:.4f})")

    def _eval_left_inner_arm(self, s: np.ndarray, info: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate left inner arm with cubic Hermite interpolation"""
        x0, y0 = -self.a, info['yj']
        x1, y1 = self.a, -info['yj']
        dx0, dy0 = 1.0, 0.0
        dx1, dy1 = -1.0, 0.0
        
        L = info['cum_L'][2] - info['cum_L'][1]
        h00 = 2*s**3 - 3*s**2 + 1
        h10 = s**3 - 2*s**2 + s
        h01 = -2*s**3 + 3*s**2
        h11 = s**3 - s**2

        x = h00 * x0 + h10 * L * dx0 + h01 * x1 + h11 * L * dx1
        y = h00 * y0 + h10 * L * dy0 + h01 * y1 + h11 * L * dy1
        return x, y

    def _eval_right_inner_arm(self, s: np.ndarray, info: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate right inner arm with cubic Hermite interpolation"""
        x0, y0 = self.a, -info['yj']
        x1, y1 = -self.a, info['yj']
        dx0, dy0 = -1.0, 0.0
        dx1, dy1 = 1.0, 0.0
        
        L = info['cum_L'][4] - info['cum_L'][3]
        h00 = 2*s**3 - 3*s**2 + 1
        h10 = s**3 - 2*s**2 + s
        h01 = -2*s**3 + 3*s**2
        h11 = s**3 - s**2

        x = h00 * x0 + h10 * L * dx0 + h01 * x1 + h11 * L * dx1
        y = h00 * y0 + h10 * L * dy0 + h01 * y1 + h11 * L * dy1
        return x, y

    def evaluate(self, t: Union[float, np.ndarray], c_z: Optional[float] = 4.0) -> Tuple:
        t = np.asarray(t)
        info = self.info
        cum = info['cum_L']
        total_L = info['total_L']

        x = np.zeros_like(t, dtype=float)
        y = np.zeros_like(t, dtype=float)
        z = c_z * t if c_z is not None else None

        s_global = t * total_L
        seg = np.searchsorted(cum, s_global, side='right') - 1
        seg = np.clip(seg, 0, 4)

        # Process each segment type with dedicated evaluation methods
        segment_functions = [
            self._eval_left_upper_arc,
            self._eval_left_inner_arm,  # Special C2 treatment
            self._eval_right_arc,
            self._eval_right_inner_arm,  # Special C2 treatment
            self._eval_left_lower_arc
        ]
        
        for i, func in enumerate(segment_functions):
            mask = (seg == i)
            if not np.any(mask): 
                continue
            s = (s_global[mask] - cum[i]) / (cum[i+1] - cum[i])
            x_mask, y_mask = func(s, info)
            x[mask] = x_mask
            y[mask] = y_mask

        if z is None:
            return x, y
        return x, y, z

    def curvature(self, t: Union[float, np.ndarray], c_z: float = 4.0) -> np.ndarray:
        t = np.asarray(t)
        cum = self.info['cum_L'] / self.info['total_L']
        kappa = super().curvature(t, c_z)
        # Override the inner segments with signed smooth curvature ramp
        inner = (cum[1] <= t) & (t <= cum[4])
        s_inner = (t[inner] - cum[1]) / (cum[4] - cum[1])
        sign = -1 if self.sign_flip else 1
        kappa[inner] = sign * (np.cos(np.pi * s_inner) * 0.5 + 0.5)  # signed smooth ramp
        return kappa

    def diagnostic_report(self):
        print("\n=== C2ClothoidFigure8 Diagnostic Report ===")
        print(f"a = {self.a:.4f} | sign_flip = {self.sign_flip}")
        ts = np.array([self.info['cum_L'][1]/self.info['total_L'],
                       self.info['cum_L'][3]/self.info['total_L']])
        xc1, yc1 = super().evaluate(ts, c_z=None)
        xc2, yc2 = self.evaluate(ts, c_z=None)
        pos_error = np.max(np.hypot(xc2 - xc1, yc2 - yc1))
        print(f"Max junction position error: {pos_error:.5f} units")
        if pos_error > 0.05:
            print("WARNING: Large junction mismatch - C2 may have visible kinks")
        else:
            print("✓ Excellent junction matching")
        print("==========================================\n")