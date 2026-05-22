"""
=====================================================================
  YIELD CURVE MODELING & ANALYSIS
  Author : Tanusree Saha
  Purpose: Quantitative Finance Portfolio Project
=====================================================================

WHAT THIS PROJECT DOES:
    1. Bootstraps a zero-coupon yield curve from coupon bond prices
    2. Fits the Nelson-Siegel model to the yield curve
    3. Models three classic curve shapes:
         - Normal (upward sloping)
         - Inverted (downward sloping — recession signal)
         - Humped (short-term rates peak then fall)
    4. Computes forward rates from the zero curve
    5. Prices bonds using the bootstrapped zero curve
    6. Analyses duration and convexity (interest rate risk)
    7. Visualises the full term structure analysis

FINANCIAL CONCEPTS COVERED:
    - Zero-coupon rates vs coupon bond yields
    - Bootstrapping: extracting zero rates from market prices
    - Nelson-Siegel parametric yield curve model
    - Forward rates: market's implied future short rates
    - Duration & Convexity: bond price sensitivity to rates
    - Yield curve shapes and their economic meaning
    - Spot rate vs par rate vs forward rate

WHY THIS MATTERS FOR RISK MANAGEMENT:
    The yield curve is the backbone of fixed income markets.
    Every interest rate product — bonds, swaps, mortgages,
    structured products — is priced off it. At banks like UBS,
    the rates desk marks hundreds of billions of positions to
    a live yield curve daily. Understanding its construction
    and dynamics is core to any rates or risk role.
=====================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit, brentq
from scipy.interpolate import CubicSpline
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
#  1. SYNTHETIC MARKET DATA
# ─────────────────────────────────────────────

def create_market_data():
    """
    Synthetic US Treasury market data.

    In practice you would pull this from Bloomberg or the
    US Treasury website (https://home.treasury.gov/resource-center/data-chart-center/interest-rates).

    Coupon bonds are quoted as (maturity_years, coupon_rate, price).
    Par = 100. Coupons paid semi-annually.
    """
    # (maturity in years, annual coupon rate %, dirty price)
    bonds = pd.DataFrame({
        'maturity'   : [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0],
        'coupon_rate': [0.00, 0.00, 2.50, 3.00, 3.25, 3.50, 3.75, 4.00,  4.25,  4.50],
        'price'      : [98.80, 97.55, 97.60, 95.30, 93.10, 88.20, 83.90, 78.40, 63.50, 54.20],
        'face_value' : [100]*10
    })
    return bonds


# ─────────────────────────────────────────────
#  2. YIELD-TO-MATURITY
# ─────────────────────────────────────────────

def compute_ytm(price, face, coupon_rate, maturity, freq=2):
    """
    Compute Yield-to-Maturity (YTM) numerically via Brent's method.

    YTM is the single discount rate that equates the present value
    of all cash flows to the current market price.

    sum[ C/freq / (1 + y/freq)^t ] + Face / (1 + y/freq)^n  =  Price

    Parameters
    ----------
    price       : float – Dirty price of the bond
    face        : float – Face (par) value
    coupon_rate : float – Annual coupon rate (%)
    maturity    : float – Years to maturity
    freq        : int   – Coupon frequency per year (2 = semi-annual)
    """
    coupon = face * (coupon_rate / 100) / freq
    n_periods = int(maturity * freq)

    if n_periods == 0:
        # Zero coupon / discount instrument
        return (face / price) ** (1 / maturity) - 1

    def price_from_ytm(y):
        periods = np.arange(1, n_periods + 1)
        pv_coupons  = np.sum(coupon / (1 + y/freq)**periods)
        pv_face     = face   / (1 + y/freq)**n_periods
        return pv_coupons + pv_face - price

    try:
        ytm = brentq(price_from_ytm, -0.5, 5.0, xtol=1e-8)
    except ValueError:
        ytm = np.nan
    return ytm


# ─────────────────────────────────────────────
#  3. BOOTSTRAPPING THE ZERO CURVE
# ─────────────────────────────────────────────

def bootstrap_zero_curve(bonds):
    """
    Bootstrap zero-coupon rates from coupon bond prices.

    CONCEPT:
        A coupon bond is just a portfolio of zero-coupon bonds.
        Once we know the zero rate for all maturities up to T-1,
        we can strip out the last cash flow to find the zero rate at T.

    ALGORITHM (iterative stripping):
        1. Short maturities (≤ 1 year): compute zero rate directly from price.
        2. For each longer maturity:
           a. Discount all coupon cash flows at known zero rates.
           b. The remaining price is the PV of the final cash flow.
           c. Solve for the zero rate that gives this PV.

    Returns
    -------
    zero_curve : pd.DataFrame with columns [maturity, zero_rate, price]
    """
    zero_rates = {}   # {maturity: zero_rate}
    results    = []

    bonds_sorted = bonds.sort_values('maturity').reset_index(drop=True)

    for _, bond in bonds_sorted.iterrows():
        T      = bond['maturity']
        coupon = bond['face_value'] * (bond['coupon_rate'] / 100) / 2
        price  = bond['price']
        face   = bond['face_value']
        n      = int(T * 2)   # semi-annual periods

        if T <= 1.0 or bond['coupon_rate'] == 0:
            # Short end: direct calculation
            if bond['coupon_rate'] == 0:
                z = -np.log(price / face) / T   # Continuous compounding
            else:
                # Simple bootstrap for ≤1yr coupon bond
                ytm = compute_ytm(price, face, bond['coupon_rate'], T)
                z   = ytm  # Close enough for short maturities

            zero_rates[T] = z
            results.append({'maturity': T, 'zero_rate': z * 100,
                            'ytm': z * 100, 'price': price})
            continue

        # Discount known coupon cash flows using existing zero rates
        pv_known_coupons = 0.0
        coupon_times = np.arange(0.5, T, 0.5)

        for t in coupon_times:
            # Interpolate zero rate at this tenor
            known_mats  = sorted(zero_rates.keys())
            known_rates = [zero_rates[m] for m in known_mats]

            if t <= known_mats[0]:
                z_t = known_rates[0]
            elif t >= known_mats[-1]:
                z_t = known_rates[-1]
            else:
                z_t = np.interp(t, known_mats, known_rates)

            pv_known_coupons += coupon * np.exp(-z_t * t)

        # Final cash flow: coupon + face
        final_cf = coupon + face
        pv_final = price - pv_known_coupons

        if pv_final <= 0:
            continue

        # Solve: pv_final = final_cf * exp(-z_T * T)
        z_T = -np.log(pv_final / final_cf) / T
        zero_rates[T] = z_T

        ytm = compute_ytm(price, face, bond['coupon_rate'], T)
        results.append({'maturity': T, 'zero_rate': z_T * 100,
                        'ytm': ytm * 100, 'price': price})

    return pd.DataFrame(results).sort_values('maturity').reset_index(drop=True)


# ─────────────────────────────────────────────
#  4. NELSON-SIEGEL MODEL
# ─────────────────────────────────────────────

def nelson_siegel(t, beta0, beta1, beta2, tau):
    """
    Nelson-Siegel parametric yield curve model.

    The most widely used model by central banks and practitioners.
    Decomposes the yield curve into three components:

        y(t) = β0  +  β1 · (1 - e^(-t/τ)) / (t/τ)
                   +  β2 · [(1 - e^(-t/τ)) / (t/τ) - e^(-t/τ)]

    Parameters
    ----------
    β0  : long-run level (as t→∞, y→β0)
    β1  : short-term component (slope at t=0)
    β2  : medium-term component (hump/trough shape)
    τ   : decay factor (controls where the hump occurs)

    Economic interpretation:
        β0        = long-run rate (monetary policy anchor)
        β0 + β1   = instantaneous short rate
        β2 > 0    = humped curve; β2 < 0 = inverted hump
    """
    x = t / tau
    factor1 = (1 - np.exp(-x)) / x
    factor2 = factor1 - np.exp(-x)
    return beta0 + beta1 * factor1 + beta2 * factor2


def fit_nelson_siegel(zero_curve):
    """Fit Nelson-Siegel to bootstrapped zero rates."""
    maturities = zero_curve['maturity'].values
    rates      = zero_curve['zero_rate'].values

    try:
        popt, _ = curve_fit(
            nelson_siegel, maturities, rates,
            p0=[4.0, -1.5, 1.5, 2.0],
            bounds=([0, -10, -10, 0.1], [15, 10, 10, 30]),
            maxfev=10000
        )
        beta0, beta1, beta2, tau = popt
    except Exception:
        beta0, beta1, beta2, tau = rates[-1], rates[0]-rates[-1], 0, 2.0

    return beta0, beta1, beta2, tau


# ─────────────────────────────────────────────
#  5. FORWARD RATES
# ─────────────────────────────────────────────

def compute_forward_rates(maturities, zero_rates_pct):
    """
    Compute instantaneous forward rates from zero rates.

    The forward rate f(t) is the rate implied for an infinitesimally
    short loan starting at time t.

    Using continuous compounding:
        f(t) = z(t) + t · dz/dt

    The forward curve tells us what the market implies about
    future short-term interest rates — key for:
        - Swap pricing (fixed leg = integral of forward rates)
        - Central bank policy expectations
        - Relative value trades on the curve
    """
    z   = zero_rates_pct / 100
    cs  = CubicSpline(maturities, z)
    t   = np.linspace(maturities[0], maturities[-1], 300)
    dz  = cs(t, 1)   # First derivative
    fwd = cs(t) + t * dz
    return t, fwd * 100


# ─────────────────────────────────────────────
#  6. BOND PRICING USING ZERO CURVE
# ─────────────────────────────────────────────

def price_bond_zero_curve(face, coupon_rate, maturity, zero_curve_df):
    """
    Price a bond using the bootstrapped zero curve.

    This is more accurate than using a flat YTM because it discounts
    each cash flow at the appropriate zero rate for its maturity.

    Price = Σ [ CF_t × e^(-z(t) × t) ]
    """
    mats  = zero_curve_df['maturity'].values
    rates = zero_curve_df['zero_rate'].values / 100

    coupon    = face * (coupon_rate / 100) / 2
    cf_times  = np.arange(0.5, maturity + 0.01, 0.5)
    price     = 0.0

    for t in cf_times:
        z_t = np.interp(t, mats, rates)
        cf  = coupon if t < maturity else coupon + face
        price += cf * np.exp(-z_t * t)

    return price


# ─────────────────────────────────────────────
#  7. DURATION & CONVEXITY
# ─────────────────────────────────────────────

def duration_convexity(face, coupon_rate, maturity, ytm, freq=2):
    """
    Compute Modified Duration and Convexity.

    DURATION (Macaulay):
        Weighted average time to receive cash flows.
        Duration ≈ % price change per 1% change in yield.

    MODIFIED DURATION:
        D_mod = D_mac / (1 + y/freq)
        For a 1bp (0.01%) yield change:
            ΔP ≈ -D_mod × P × Δy

    CONVEXITY:
        Second-order correction. Always positive for regular bonds.
        ΔP ≈ -D_mod × P × Δy + 0.5 × Convexity × P × Δy²

    High convexity is desirable — the bond gains more when
    yields fall than it loses when yields rise.
    """
    coupon    = face * (coupon_rate / 100) / freq
    n_periods = int(maturity * freq)
    y_per     = ytm / freq

    periods   = np.arange(1, n_periods + 1)
    cfs       = np.full(n_periods, coupon)
    cfs[-1]  += face

    pv_cfs = cfs / (1 + y_per)**periods
    price  = pv_cfs.sum()

    # Macaulay Duration
    d_mac = np.sum(periods * pv_cfs) / (price * freq)

    # Modified Duration
    d_mod = d_mac / (1 + y_per)

    # Convexity
    t_sq   = periods * (periods + 1)
    convex = np.sum(t_sq * pv_cfs) / (price * (1 + y_per)**2 * freq**2)

    return {'Macaulay Duration': round(d_mac, 4),
            'Modified Duration': round(d_mod, 4),
            'Convexity'        : round(convex, 4),
            'Price'            : round(price,  4)}


# ─────────────────────────────────────────────
#  8. CURVE SHAPES
# ─────────────────────────────────────────────

def generate_curve_scenarios():
    """
    Three classic yield curve shapes and their economic meaning.
    """
    t = np.linspace(0.25, 30, 200)
    scenarios = {
        'Normal (Upward Sloping)': {
            'rates'  : nelson_siegel(t, 4.5, -1.8, 0.5, 2.0),
            'color'  : '#00C896',
            'meaning': 'Economy growing normally. Long rates > short rates.'
        },
        'Inverted (Recession Signal)': {
            'rates'  : nelson_siegel(t, 3.5, 2.0, -0.5, 1.5),
            'color'  : '#FF4444',
            'meaning': 'Short rates > long rates. Historically predicts recession.'
        },
        'Humped': {
            'rates'  : nelson_siegel(t, 4.0, -0.5, 2.5, 3.0),
            'color'  : '#FFB347',
            'meaning': 'Medium-term rates highest. Transition period in policy.'
        },
    }
    return t, scenarios


# ─────────────────────────────────────────────
#  9. VISUALISATION
# ─────────────────────────────────────────────

def plot_yield_curve_analysis(zero_curve, ns_params, bonds):
    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor('#0D1117')
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.48, wspace=0.35)

    BG     = '#0D1117'
    PANEL  = '#161B22'
    GRID_C = '#2A2A3A'
    TEXT_C = '#E0E0E0'
    GREEN  = '#00C896'
    RED    = '#FF4444'
    AMBER  = '#FFB347'
    BLUE   = '#4A9EFF'
    PURPLE = '#C084FC'
    GOLD   = '#FFD700'

    t_smooth = np.linspace(0.25, 30, 300)
    b0, b1, b2, tau = ns_params
    ns_smooth = nelson_siegel(t_smooth, b0, b1, b2, tau)

    # ── Panel 1: Bootstrapped Zero Curve + Nelson-Siegel fit ──
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(PANEL)
    ax1.scatter(zero_curve['maturity'], zero_curve['zero_rate'],
                color=GOLD, s=60, zorder=5, label='Bootstrapped Zero Rates', marker='D')
    ax1.scatter(zero_curve['maturity'], zero_curve['ytm'],
                color=BLUE, s=40, zorder=5, label='YTM (Par Rates)', marker='o', alpha=0.7)
    ax1.plot(t_smooth, ns_smooth, color=GREEN, linewidth=2,
             label=f'Nelson-Siegel Fit\nβ0={b0:.2f}, β1={b1:.2f}, β2={b2:.2f}, τ={tau:.2f}')
    ax1.set_title('Bootstrapped Zero Curve & Nelson-Siegel Model',
                  color=TEXT_C, fontsize=10, fontweight='bold')
    ax1.set_xlabel('Maturity (years)', color=TEXT_C, fontsize=9)
    ax1.set_ylabel('Rate (%)', color=TEXT_C, fontsize=9)
    ax1.tick_params(colors=TEXT_C)
    ax1.grid(True, color=GRID_C, linewidth=0.5)
    ax1.legend(fontsize=7.5, facecolor=PANEL, labelcolor=TEXT_C)
    for s in ax1.spines.values(): s.set_color(GRID_C)

    # ── Panel 2: Curve Scenarios ──
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(PANEL)
    t_sc, scenarios = generate_curve_scenarios()
    for name, sc in scenarios.items():
        ax2.plot(t_sc, sc['rates'], color=sc['color'], linewidth=2, label=name)
    ax2.set_title('Yield Curve Shapes & Economic Signals',
                  color=TEXT_C, fontsize=10, fontweight='bold')
    ax2.set_xlabel('Maturity (years)', color=TEXT_C, fontsize=9)
    ax2.set_ylabel('Yield (%)', color=TEXT_C, fontsize=9)
    ax2.tick_params(colors=TEXT_C)
    ax2.grid(True, color=GRID_C, linewidth=0.5)
    ax2.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT_C)
    for s in ax2.spines.values(): s.set_color(GRID_C)

    # ── Panel 3: Forward Rates ──
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor(PANEL)
    fwd_t, fwd_r = compute_forward_rates(zero_curve['maturity'].values,
                                          zero_curve['zero_rate'].values)
    ax3.plot(t_smooth, ns_smooth, color=GOLD, linewidth=1.5,
             linestyle='--', label='Zero Rate (spot)', alpha=0.8)
    ax3.plot(fwd_t, fwd_r, color=PURPLE, linewidth=2, label='Instantaneous Forward Rate')
    ax3.set_title('Spot (Zero) Rates vs Forward Rates',
                  color=TEXT_C, fontsize=10, fontweight='bold')
    ax3.set_xlabel('Maturity (years)', color=TEXT_C, fontsize=9)
    ax3.set_ylabel('Rate (%)', color=TEXT_C, fontsize=9)
    ax3.tick_params(colors=TEXT_C)
    ax3.grid(True, color=GRID_C, linewidth=0.5)
    ax3.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT_C)
    for s in ax3.spines.values(): s.set_color(GRID_C)

    # ── Panel 4: Discount Factor Curve ──
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(PANEL)
    disc = np.exp(-ns_smooth / 100 * t_smooth)
    ax4.plot(t_smooth, disc, color=BLUE, linewidth=2, label='Discount Factor P(0,T)')
    ax4.fill_between(t_smooth, disc, alpha=0.15, color=BLUE)
    ax4.set_title('Discount Factor Curve  P(0,T) = e^(-z(T)·T)',
                  color=TEXT_C, fontsize=10, fontweight='bold')
    ax4.set_xlabel('Maturity (years)', color=TEXT_C, fontsize=9)
    ax4.set_ylabel('Discount Factor', color=TEXT_C, fontsize=9)
    ax4.tick_params(colors=TEXT_C)
    ax4.grid(True, color=GRID_C, linewidth=0.5)
    ax4.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT_C)
    for s in ax4.spines.values(): s.set_color(GRID_C)

    # ── Panel 5: Duration vs Maturity ──
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.set_facecolor(PANEL)
    mats_plot = [1, 2, 3, 5, 7, 10, 20, 30]
    coup_rates = [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]
    ytms_ref   = [np.interp(m, zero_curve['maturity'], zero_curve['zero_rate'])/100
                  for m in mats_plot]
    durs   = [duration_convexity(100, c, m, y)['Modified Duration']
              for m, c, y in zip(mats_plot, coup_rates, ytms_ref)]
    convex = [duration_convexity(100, c, m, y)['Convexity']
              for m, c, y in zip(mats_plot, coup_rates, ytms_ref)]
    ax5b = ax5.twinx()
    ax5.bar(mats_plot, durs, width=1.2, color=GREEN, alpha=0.7, label='Modified Duration')
    ax5b.plot(mats_plot, convex, color=AMBER, linewidth=2, marker='o', ms=5, label='Convexity')
    ax5.set_title('Modified Duration & Convexity vs Maturity (3% coupon bond)',
                  color=TEXT_C, fontsize=10, fontweight='bold')
    ax5.set_xlabel('Maturity (years)', color=TEXT_C, fontsize=9)
    ax5.set_ylabel('Modified Duration (years)', color=TEXT_C, fontsize=9)
    ax5b.set_ylabel('Convexity', color=AMBER, fontsize=9)
    ax5.tick_params(colors=TEXT_C)
    ax5b.tick_params(colors=AMBER)
    ax5.grid(True, color=GRID_C, linewidth=0.5, axis='y')
    lines1, labels1 = ax5.get_legend_handles_labels()
    lines2, labels2 = ax5b.get_legend_handles_labels()
    ax5.legend(lines1+lines2, labels1+labels2, fontsize=8, facecolor=PANEL, labelcolor=TEXT_C)
    for s in ax5.spines.values(): s.set_color(GRID_C)

    # ── Panel 6: Bond Pricing vs Yield shock ──
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.set_facecolor(PANEL)
    yield_shocks = np.linspace(-0.03, 0.03, 100)
    for mat, coup, color, lbl in [(2, 3, BLUE, '2yr 3% bond'),
                                   (10, 4, GREEN, '10yr 4% bond'),
                                   (30, 4.5, RED, '30yr 4.5% bond')]:
        ytm_base = np.interp(mat, zero_curve['maturity'],
                             zero_curve['zero_rate']) / 100
        prices = [price_bond_zero_curve(100, coup, mat,
                   zero_curve.assign(zero_rate=zero_curve['zero_rate'] + shock*100))
                  for shock in yield_shocks]
        ax6.plot(yield_shocks * 100, prices, color=color, linewidth=2, label=lbl)
    ax6.axvline(0, color=GRID_C, linewidth=1, linestyle='--')
    ax6.axhline(100, color=GRID_C, linewidth=0.8, linestyle=':')
    ax6.set_title('Bond Price vs Parallel Yield Curve Shift',
                  color=TEXT_C, fontsize=10, fontweight='bold')
    ax6.set_xlabel('Yield Curve Shift (%)', color=TEXT_C, fontsize=9)
    ax6.set_ylabel('Bond Price ($)', color=TEXT_C, fontsize=9)
    ax6.tick_params(colors=TEXT_C)
    ax6.grid(True, color=GRID_C, linewidth=0.5)
    ax6.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT_C)
    for s in ax6.spines.values(): s.set_color(GRID_C)

    fig.suptitle('Yield Curve Modeling: Bootstrapping, Nelson-Siegel & Interest Rate Risk',
                 color=TEXT_C, fontsize=13, fontweight='bold', y=1.01)

    plt.savefig('/mnt/user-data/outputs/yield_curve_analysis.png',
                dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print("Chart saved.")


# ─────────────────────────────────────────────
#  10. MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 65)
    print("  YIELD CURVE MODELING — Bootstrapping & Nelson-Siegel")
    print("=" * 65)

    bonds = create_market_data()
    print(f"\n── Input: {len(bonds)} Treasury bonds ──")
    print(bonds[['maturity','coupon_rate','price']].to_string(index=False))

    # ── Bootstrap ──
    print("\n── Bootstrapping Zero Curve ──")
    zero_curve = bootstrap_zero_curve(bonds)
    print(f"\n  {'Maturity':>10}  {'Zero Rate':>12}  {'YTM':>10}  {'Spread (Z-YTM)':>16}")
    for _, row in zero_curve.iterrows():
        spread = row['zero_rate'] - row['ytm']
        print(f"  {row['maturity']:>8.2f}yr  {row['zero_rate']:>10.4f}%  "
              f"{row['ytm']:>8.4f}%  {spread:>+14.4f}%")

    # ── Nelson-Siegel Fit ──
    print("\n── Nelson-Siegel Model Fit ──")
    ns_params = fit_nelson_siegel(zero_curve)
    b0, b1, b2, tau = ns_params
    print(f"  β0 (long-run level)    : {b0:.4f}%")
    print(f"  β1 (short-term slope)  : {b1:.4f}%")
    print(f"  β2 (medium-term hump)  : {b2:.4f}%")
    print(f"  τ  (decay factor)      : {tau:.4f}")
    print(f"  Instantaneous short rate (β0+β1): {b0+b1:.4f}%")
    print(f"  Long-run rate (β0)     : {b0:.4f}%")

    # Goodness of fit
    fitted = [nelson_siegel(m, b0, b1, b2, tau) for m in zero_curve['maturity']]
    rmse   = np.sqrt(np.mean((np.array(fitted) - zero_curve['zero_rate'].values)**2))
    print(f"  RMSE of fit            : {rmse:.6f}%")

    # ── Forward Rates ──
    print("\n── Selected Forward Rates ──")
    fwd_t, fwd_r = compute_forward_rates(zero_curve['maturity'].values,
                                          zero_curve['zero_rate'].values)
    for horizon in [1, 2, 5, 10, 20]:
        idx = np.argmin(np.abs(fwd_t - horizon))
        z_h = nelson_siegel(horizon, b0, b1, b2, tau)
        print(f"  {horizon:>2}yr — Zero rate: {z_h:.4f}%  |  "
              f"Forward rate: {fwd_r[idx]:.4f}%  |  "
              f"Fwd premium: {fwd_r[idx]-z_h:+.4f}%")

    # ── Bond Pricing ──
    print("\n── Bond Pricing via Zero Curve vs YTM ──")
    test_bonds = [
        (100, 4.0, 5.0,  'Treasury 4.00% 5yr'),
        (100, 4.0, 10.0, 'Treasury 4.00% 10yr'),
        (100, 5.0, 10.0, 'Corp 5.00% 10yr'),
    ]
    for face, coup, mat, name in test_bonds:
        p_zero = price_bond_zero_curve(face, coup, mat, zero_curve)
        ytm_   = compute_ytm(p_zero, face, coup, mat)
        print(f"  {name:28s}: Price = ${p_zero:.4f}  |  "
              f"Implied YTM = {ytm_*100:.4f}%")

    # ── Duration & Convexity ──
    print("\n── Duration & Convexity Analysis ──")
    print(f"  {'Bond':>25}  {'D_mac':>8}  {'D_mod':>8}  {'Convex':>8}  {'Price':>8}")
    for face, coup, mat, name in test_bonds:
        ytm_ = compute_ytm(
            price_bond_zero_curve(face, coup, mat, zero_curve), face, coup, mat)
        dc = duration_convexity(face, coup, mat, ytm_)
        print(f"  {name:>25}  {dc['Macaulay Duration']:>8.4f}  "
              f"{dc['Modified Duration']:>8.4f}  {dc['Convexity']:>8.4f}  "
              f"${dc['Price']:>7.4f}")

    # ── Rate Sensitivity ──
    print("\n── Price Sensitivity to +1bp Yield Shift (using Modified Duration) ──")
    for face, coup, mat, name in test_bonds:
        p  = price_bond_zero_curve(face, coup, mat, zero_curve)
        ytm_ = compute_ytm(p, face, coup, mat)
        dc = duration_convexity(face, coup, mat, ytm_)
        dv01 = dc['Modified Duration'] * p * 0.0001   # DV01: $ change per 1bp
        print(f"  {name:28s}: DV01 = ${dv01:.4f}  "
              f"(${dv01*100:.2f} per $100 face, per 1bp)")

    # ── Curve Shapes Summary ──
    print("\n── Yield Curve Shape Economic Interpretation ──")
    shapes = {
        'Normal (Upward)' : 'Long rates > short rates. Growth expected, inflation priced in.',
        'Inverted'        : 'Short rates > long rates. Market expects rate cuts = recession signal.',
        'Flat'            : 'Transition between normal and inverted. Policy uncertainty.',
        'Humped'          : 'Medium-term rates peak. Short-end tightening, long-end anchored.',
    }
    for shape, meaning in shapes.items():
        print(f"  {shape:20s}: {meaning}")

    print("\n── Generating Charts ──")
    plot_yield_curve_analysis(zero_curve, ns_params, bonds)
    print("\nDone. Chart saved to outputs folder.")
