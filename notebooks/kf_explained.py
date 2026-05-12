import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.widgets import Slider

# ─────────────────────────────────────────────
# TRUE SYSTEM
# ─────────────────────────────────────────────

def true_bias(t):
    return np.array([0.3 * np.sin(0.2 * t), 0.2 * np.cos(0.15 * t)])

def true_dynamics(x, t):
    X, Y = x
    b = true_bias(t)
    return np.array([np.sin(Y) * np.sin(t) + b[0],
                     np.sin(X) * np.sin(t) + b[1]])

def simulate_truth(x0, T, dt, sigma_q):
    times = np.arange(0, T, dt)
    xs = [np.array(x0, dtype=float)]
    for i in range(len(times) - 1):
        u = np.random.normal(0, sigma_q, size=2)
        xs.append(xs[-1] + (true_dynamics(xs[-1], dt * i) + u) * dt)
    return np.array(xs), times

# ─────────────────────────────────────────────
# AUGMENTED EKF  —  state = [X, Y, bx, by]
# ─────────────────────────────────────────────

def ekf_dynamics(x_hat, t):
    X, Y, bx, by = x_hat
    return np.array([
        np.sin(Y) * np.sin(t) + bx,
        np.sin(X) * np.sin(t) + by,
        0.0, 0.0
    ])

def jacobian(x_hat, t):
    X, Y, bx, by = x_hat
    return np.array([
        [0,                      np.cos(Y) * np.sin(t), 1, 0],
        [np.cos(X) * np.sin(t), 0,                      0, 1],
        [0,                      0,                      0, 0],
        [0,                      0,                      0, 0]
    ])

def ekf_predict(x_hat, P, t, dt, Q):
    F = jacobian(x_hat, t)
    return x_hat + ekf_dynamics(x_hat, t) * dt, P + (F @ P + P @ F.T + Q) * dt

def ekf_update(x_hat, P, y):
    """
    Perfect observation update (R = 0).

    State partitioned as:
        x_hat = [pos (2,), bias (2,)]
        P     = [[P_XX (2x2),  P_Xb (2x2)],
                 [P_bX (2x2),  P_bb (2x2)]]

    Innovation:
        nu = y - pos

    State update:
        pos_new  = y                              (snap exactly to observation)
        bias_new = bias + P_bX @ inv(P_XX) @ nu  (bias updated via correlation)

    Covariance update:
        P_XX_new = 0                              (perfect position knowledge)
        P_Xb_new = 0
        P_bX_new = 0
        P_bb_new = P_bb - P_bX @ inv(P_XX) @ P_Xb  (Schur complement)
    """
    # Extract blocks
    P_XX = P[:2, :2]
    P_Xb = P[:2, 2:]
    P_bX = P[2:, :2]
    P_bb = P[2:, 2:]

    pos  = x_hat[:2]
    bias = x_hat[2:]

    # Innovation
    nu = y - pos

    # Gain for bias block only
    P_XX_inv = np.linalg.inv(P_XX)

    # State update
    pos_new  = y
    bias_new = bias + P_bX @ P_XX_inv @ nu

    # Covariance update  (Schur complement for bias block)
    P_bb_new = P_bb - P_bX @ P_XX_inv @ P_Xb

    # Assemble updated state and covariance
    x_new = np.concatenate([pos_new, bias_new])
    P_new = np.zeros((4, 4))
    P_new[2:, 2:] = P_bb_new          # bias uncertainty reduced by observation
    # position blocks are exactly zero — but add tiny epsilon for numerical stability
    P_new[:2, :2] = np.eye(2) * 1e-10

    return x_new, P_new

# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────

np.random.seed(42)
T, dt, sigma_q, obs_every = 60.0, 0.3, 0.1, 10
sigma_bias = 0.2

Q = np.diag([sigma_q**2, sigma_q**2, sigma_bias**2, sigma_bias**2])

x0_true = [0.5, 0.5]
truth, times = simulate_truth(x0_true, T, dt, sigma_q)

x_hat = np.array([0.5, 0.5, 0.0, 0.0])
P = np.eye(4) * 1.0

estimates, covariances, observations, obs_steps = [x_hat.copy()], [P.copy()], [], []

for i in range(1, len(times)):
    x_hat, P = ekf_predict(x_hat, P, times[i-1], dt, Q)
    if i % obs_every == 0:
        y = truth[i]          # perfect observation — no noise added
        observations.append(y)
        obs_steps.append(i)
        x_hat, P = ekf_update(x_hat, P, y)
    estimates.append(x_hat.copy())
    covariances.append(P.copy())

estimates   = np.array(estimates)
true_biases = np.array([true_bias(t) for t in times])
traces      = [np.trace(c) for c in covariances]

# ─────────────────────────────────────────────
# LAYOUT
# ─────────────────────────────────────────────

fig = plt.figure(figsize=(18, 7))
fig.patch.set_facecolor('#0d1117')

ax    = fig.add_axes([0.04, 0.15, 0.28, 0.78])
ax2   = fig.add_axes([0.37, 0.15, 0.28, 0.78])
ax3   = fig.add_axes([0.70, 0.15, 0.28, 0.78])
ax_sl = fig.add_axes([0.10, 0.04, 0.80, 0.03])

for a in (ax, ax2, ax3):
    a.set_facecolor('#0d1117')
    a.tick_params(colors='white')
    for sp in a.spines.values(): sp.set_color('#333')

lim = max(np.abs(truth).max(), np.abs(estimates).max()) * 1.1
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.set_xlabel('X', color='white'); ax.set_ylabel('Y', color='white')
ax.set_title('Trajectory', color='white')
ax.plot(truth[:, 0], truth[:, 1], color='white', lw=0.5, alpha=0.2)

ax2.set_xlim(0, T); ax2.set_ylim(0, max(traces) * 1.1)
ax2.set_xlabel('Time', color='white'); ax2.set_ylabel('Trace(P)', color='white')
ax2.set_title('Total uncertainty', color='white')
ax2.plot(times, traces, color='#333', lw=0.8)

bias_max = np.max(np.abs(true_biases)) * 1.8
ax3.set_xlim(0, T); ax3.set_ylim(-bias_max, bias_max)
ax3.set_xlabel('Time', color='white'); ax3.set_ylabel('Bias', color='white')
ax3.set_title('Bias learning', color='white')
ax3.plot(times, true_biases[:, 0], color='cyan',    lw=1.0, alpha=0.3, ls='--', label='True $b_X$')
ax3.plot(times, true_biases[:, 1], color='magenta', lw=1.0, alpha=0.3, ls='--', label='True $b_Y$')
ax3.axhline(0, color='#444', lw=0.5)

# Dynamic elements
true_line,  = ax.plot([], [], color='white', lw=1.2, alpha=0.7, label='Truth')
est_line,   = ax.plot([], [], color='cyan',  lw=1.2, alpha=0.9, ls='--', label='EKF estimate')
obs_scatter = ax.scatter([], [], color='yellow', s=30, zorder=5, label='Observation')
true_dot,   = ax.plot([], [], 'o', color='white', ms=6, zorder=7)
est_dot,    = ax.plot([], [], 'o', color='cyan',  ms=6, zorder=7)
ellipse_patch = Ellipse((0,0), 0, 0, alpha=0.15, facecolor='cyan', edgecolor='cyan', lw=1.2)
ax.add_patch(ellipse_patch)
ax.legend(facecolor='#1a1a2e', labelcolor='white', framealpha=0.8, loc='upper right', fontsize=8)

trace_line, = ax2.plot([], [], color='cyan', lw=1.5)
vline2 = ax2.axvline(0, color='white', alpha=0.5, lw=1.0)

bx_line, = ax3.plot([], [], color='cyan',    lw=1.5, label='Est. $b_X$')
by_line, = ax3.plot([], [], color='magenta', lw=1.5, label='Est. $b_Y$')
vline3   = ax3.axvline(0, color='white', alpha=0.5, lw=1.0)
ax3.legend(facecolor='#1a1a2e', labelcolor='white', framealpha=0.8, fontsize=8)

time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='white', fontsize=9)

slider = Slider(ax_sl, 'Step', 0, len(times)-1, valinit=0, valstep=1,
                color='cyan', track_color='#222')
ax_sl.set_facecolor('#0d1117')
slider.label.set_color('white'); slider.valtext.set_color('white')

def update_ellipse(mean, cov):
    block = cov[:2, :2]
    vals, vecs = np.linalg.eigh(block)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h = 2 * 2.0 * np.sqrt(np.abs(vals))
    ellipse_patch.set_center(mean[:2])
    ellipse_patch.width = w; ellipse_patch.height = h; ellipse_patch.angle = angle

def update(val):
    i = int(slider.val)
    true_line.set_data(truth[:i+1, 0],     truth[:i+1, 1])
    est_line.set_data( estimates[:i+1, 0], estimates[:i+1, 1])
    true_dot.set_data([truth[i, 0]],     [truth[i, 1]])
    est_dot.set_data( [estimates[i, 0]], [estimates[i, 1]])
    update_ellipse(estimates[i], covariances[i])

    vis = [(observations[k][0], observations[k][1])
           for k, s in enumerate(obs_steps) if s <= i]
    obs_scatter.set_offsets(np.array(vis) if vis else np.empty((0, 2)))

    trace_line.set_data(times[:i+1], traces[:i+1])
    bx_line.set_data(times[:i+1], estimates[:i+1, 2])
    by_line.set_data(times[:i+1], estimates[:i+1, 3])
    for vl in (vline2, vline3): vl.set_xdata([times[i], times[i]])
    time_text.set_text(f't = {times[i]:.2f}')
    fig.canvas.draw_idle()

slider.on_changed(update)
update(0)
plt.show()