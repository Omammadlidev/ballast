# Theory

Everything here is implemented in [`ballast/multibody.py`](../ballast/multibody.py)
and [`ballast/dynamics.py`](../ballast/dynamics.py), and checked by
[`tests/test_multibody.py`](../tests/test_multibody.py).

## 1. Frames and notation

The world frame is ENU with gravity along $-\hat z_w$. The body frame has its
origin $O$ at the **airframe shell's own centre of mass**, $+z$ up through the
rotor, $+x$ forward. A quaternion $q$ and its rotation matrix $R$ map body
coordinates into world coordinates, $v_w = R\,v_b$.

| symbol | meaning |
|---|---|
| $m_b,\ m_s,\ M$ | shell mass, slider mass, total |
| $\mu = m_b m_s / M$ | reduced mass |
| $J_b$ | shell inertia about its own centre of mass |
| $s(t)$ | slider position in body coordinates |
| $c$ | system centre of mass in body coordinates |
| $r_t,\ r_{cp}$ | thrust application point, centre of pressure |
| $\omega$ | body angular rate, in body coordinates |

Because $O$ is the shell's centre of mass, the shell contributes no first mass
moment about it, and

$$c = \frac{m_s}{M}\,s .$$

With $s_z$ fixed by the rail plane, $c$ is a scaled copy of the slider position.
That is the entire steering mechanism.

## 2. Why the system centre of mass is the state

The natural instinct is to integrate an airframe-fixed point. Resist it: the
system centre of mass $G$ moves *within* the body as the slider travels, so
Newton's second law about a body-fixed point picks up $\dot c$ and $\ddot c$
terms. About $G$ it stays exactly

$$M\,a_G = R\,(F_{\text{thrust}} + F_{\text{drag}}) - Mg\,\hat z_w .$$

The airframe origin — where sensors and payload actually live — is then recovered,
*including* the coupling that a naive implementation drops:

$$p_O = p_G - R\,c,\qquad
  v_O = v_G - R\big(\dot c + \omega\times c\big).$$

That $\omega\times c$ term matters: it is why a vehicle rotating at 1.5 °/s with a
12 mm centre-of-mass offset does not accumulate a phantom velocity in the
controller's feedback path.

## 3. Inertia about a moving centre of mass

Measuring both bodies from $G$ gives $\rho_b = -c$ and
$\rho_s = s - c = \frac{m_b}{M}s$. Their parallel-axis contributions are

$$m_b\big(\lVert\rho_b\rVert^2 I - \rho_b\rho_b^\top\big)
+ m_s\big(\lVert\rho_s\rVert^2 I - \rho_s\rho_s^\top\big)
= \frac{m_b m_s^2 + m_s m_b^2}{M^2}\big(s^\top s\,I - ss^\top\big),$$

and since $m_b m_s^2 + m_s m_b^2 = m_b m_s M$, the two collapse into one term:

$$\boxed{\,J_G(s) = J_b + \mu\big(s^\top s\,I - ss^\top\big)\,}$$

$$\dot J_G = \mu\big(2(s\cdot\dot s)I - \dot s s^\top - s\dot s^\top\big).$$

The test suite checks this identity against a brute-force two-body sum at 200
random slider positions, to `atol=1e-15`.

## 4. Angular momentum stored in the slider's motion

The slider moving relative to the shell carries angular momentum of its own. With
$\dot\rho_b = -\frac{m_s}{M}\dot s$ and $\dot\rho_s = \frac{m_b}{M}\dot s$,

$$h_{\mathrm{rel}} = \sum_i m_i\,\rho_i\times\dot\rho_i
= \frac{m_b m_s}{M}\,\big[-(r_b - c) + (s - c)\big]\times\dot s
= \mu\,(s\times\dot s),$$

using $r_b = 0$. So $H_G = J_G\omega + h_{\mathrm{rel}}$, and transporting the
derivative into the rotating frame,

$$\boxed{\,J_G\dot\omega = \tau_G - \omega\times\big(J_G\omega + h_{\mathrm{rel}}\big)
  - \dot J_G\,\omega - \dot h_{\mathrm{rel}}\,},
\qquad \dot h_{\mathrm{rel}} = \mu\,(s\times\ddot s).$$

Three terms here vanish in the usual simplified treatment: $\dot J_G\omega$,
$h_{\mathrm{rel}}$ inside the gyroscopic product, and $\dot h_{\mathrm{rel}}$.
All three are driven by slider *motion*, so they are largest exactly when the
controller is working hardest. Section 7 shows what dropping them costs.

## 5. Applied moments

$$\tau_G = \underbrace{(r_t - c)\times f\hat z_b}_{\text{steering}}
  + \underbrace{(r_{cp} - c)\times F_{\text{drag}}}_{\text{weathervane}}
  + \underbrace{\tau_{\text{vane}}\hat z_b}_{\text{residual reaction}}
  + \underbrace{\tau_{\text{damp}}(\omega)}_{\text{rate damping}}$$

**Steering.** With $F = f\hat z_b$ and $r_t - c = [-c_x,\,-c_y,\,r_{t,z}-c_z]^\top$,

$$(r_t - c)\times f\hat z_b = f\begin{bmatrix}-c_y\\ c_x\\ 0\end{bmatrix}.$$

The vertical lever multiplies a force parallel to itself and cancels: **rotor
mount height does not affect steering torque.** A test asserts it, comparing a
120 mm mast against a 400 mm one.

**Anti-torque vanes.** The rotor reaction torque is $-\sigma c_\tau f$. Fixed
stator vanes in the slipstream recover a fraction $\eta$ of it. Because both the
reaction and the vane force scale with thrust, $\eta$ is *thrust-independent* —
trim the vanes once on the bench and they stay trimmed through the whole flight
envelope. What is left,

$$\tau_{\text{vane}} = -\sigma c_\tau (1-\eta) f,$$

is set purely by build tolerance. At $\eta = 0.995$ it produces about 1.5 °/s of
heading drift, bounded by yaw rate damping.

**Centre of pressure.** $r_{cp}$ sits slightly *below* the centre of mass, which
makes the airframe mildly weathervane-stable — drag tilts the thrust axis into the
relative wind rather than away from it. The offset is deliberately small: the
aerodynamic moment grows as $v^2$ while the slider's authority does not, so a
large offset would produce moments the actuator cannot overpower in gusts. With
$C_dA \approx 0.02\,\text{m}^2$ and a 30 mm lever, the aerodynamic moment reaches
the slider's full authority only around 18 m/s.

## 6. Control allocation, and what is unreachable

Inverting the steering map is immediate:

$$c_x = \frac{\tau_y}{f},\qquad c_y = -\frac{\tau_x}{f},\qquad
  s_{\text{cmd}} = \frac{M}{m_s}\,c .$$

The yaw row is identically zero, so **heading is structurally uncontrollable** with
this actuator set. The controller therefore never asks for yaw: it builds its
desired attitude around the vehicle's *current* heading, so the attitude error
$e_R = \tfrac12(R_d^\top R - R^\top R_d)^\vee$ carries no yaw component that could
steal roll and pitch authority. A free-running heading then costs nothing — a test
asserts the vehicle rotates 90° while holding station to 0.1 mm.

When demand exceeds travel the allocator scales magnitude and **preserves torque
direction**, so a saturated vehicle still tilts the right way.

## 7. Where the rails go

$-\dot h_{\mathrm{rel}} = -\mu(s\times\ddot s)$ appears the instant the slider
accelerates, before the centre of mass has moved anywhere. Take a slider starting
centred on a rail plane at height $z_s$, accelerating in $+x$:

$$s\times\ddot s = \begin{bmatrix}0\\0\\z_s\end{bmatrix}\times
  \begin{bmatrix}a\\0\\0\end{bmatrix} = \begin{bmatrix}0\\ -z_s a\\ 0\end{bmatrix}
\quad\Longrightarrow\quad
-\dot h_{\mathrm{rel}} = \begin{bmatrix}0\\ \mu z_s a\\ 0\end{bmatrix}.$$

The steering torque that eventually arrives is $+y$ (positive $c_x$). So:

- $z_s < 0$ (rails **below** the centre of mass): the reaction is also $+y$. It
  **leads** the steering torque.
- $z_s > 0$ (rails **above**): the reaction is $-y$. The plant is **non-minimum
  phase** — it tilts the wrong way first.

The magnitudes are comparable, not marginal. The ratio of reaction to steering
torque at peak servo acceleration $\ddot s = \omega_n^2 \Delta s$ is

$$\frac{\mu\,|z_s|\,\omega_n^2\Delta s}{f\,(m_s/M)\,s}
\;\approx\; 1 \quad\text{for the default vehicle.}$$

Closed loop, the difference between the two layouts is 3 mm RMS versus 354 mm.
Run [`examples/06_where_to_put_the_rails.py`](../examples/06_where_to_put_the_rails.py).

## 8. Linearisation in error coordinates

The 18-element state carries a 4-element quaternion for 3 rotational degrees of
freedom, so a raw Jacobian is rank-deficient and its spectrum is polluted by the
redundant direction. `ballast/trim.py` linearises in 17 error coordinates

$$\delta x = (\delta p,\ \delta v,\ \delta\theta,\ \delta\omega,\ \delta\Omega_r,
             \ \delta s,\ \delta\dot s),$$

where attitude is perturbed multiplicatively, $q \leftarrow q\otimes\delta q(\delta\theta)$.
The resulting $A$ has exactly the structural zero modes it should — three position
integrators, the horizontal $x\leftarrow v\leftarrow\theta\leftarrow\omega$ chains,
and the free heading — and nothing spurious.

Those chains are analytically nilpotent, so a finite-difference Jacobian spreads
their eigenvalues into a small cluster around the origin. `modes()` therefore
classifies against a *relative* tolerance; an absolute one would label half the
cluster divergent. Every remaining eigenvalue matches a closed-form prediction to
$10^{-6}$ or better — see the table in the [README](../README.md#open-loop-modes).

## 9. What is not modelled

Quasi-steady aerodynamics with no rotor inflow, wake or ground effect. The slider
is a point mass with no rail friction or backlash. The rotor is a first-order lag
with no blade dynamics or aerodynamic torque variation with inflow. There are no
sensors and no estimator — the controller runs on true state. Battery voltage does
not sag, so thrust limits are constant.

An estimator is the obvious next step, and the honest one: this vehicle's attitude
loop is slow enough that estimator lag would show up directly in the numbers above.
