"""Published Monte-Carlo values transcribed verbatim from ``latex/main.tex``.

Each entry is ``(bias, se, ese, cp)``; ``None`` marks a cell the paper leaves
blank (the ``reReg-AFT`` row reports no ESE/CP). Used only for side-by-side
reporting -- nothing here is recomputed locally.

Competitor rows (``reReg-Cox``, ``reReg-AFT``, ``NPMLE``, ``Reda``) are carried
over as-is: they are not part of the local ODE module and are not re-run.
"""

# Table 1 (\label{cox_aft_table}), n=1000, 1000 reps (am.GL: 200)
TABLE1 = {
    (1, 'ODE-Cox'):   {'beta_2': (-0.002, 0.039, 0.043, 0.958),
                       'beta_3': (-0.002, 0.041, 0.043, 0.948)},
    (1, 'reReg-Cox'): {'beta_2': (-0.002, 0.039, 0.040, 0.956),
                       'beta_3': (-0.002, 0.041, 0.041, 0.941)},
    (2, 'ODE-AM'):    {'beta_2': (-0.021, 0.064, 0.063, 0.936),
                       'beta_3': (-0.022, 0.063, 0.063, 0.943)},
    (2, 'reReg-AFT'): {'beta_2': (0.005, 0.062, None, None),
                       'beta_3': (-0.000, 0.064, None, None)},
    (3, 'ODE-LT'):    {'beta_2': (0.000, 0.078, 0.072, 0.960),
                       'beta_3': (0.003, 0.088, 0.100, 0.980)},
    (3, 'NPMLE'):     {'beta_2': (0.017, 0.074, 0.078, 0.960),
                       'beta_3': (0.024, 0.101, 0.109, 0.962)},
}

# Table 2 (\label{ode_flex_table}) -- ODE-Flex, n=1000
TABLE2 = {
    1: {'beta_2': (0.005, 0.059, 0.061, 0.952),
        'beta_3': (0.006, 0.059, 0.060, 0.961)},
    2: {'beta_2': (-0.002, 0.083, 0.079, 0.933),
        'beta_3': (-0.003, 0.084, 0.079, 0.936)},
    3: {'beta_2': (0.001, 0.097, 0.092, 0.934),
        'beta_3': (0.006, 0.119, 0.116, 0.949)},
    4: {'beta_2': (-0.007, 0.071, 0.066, 0.939),
        'beta_3': (-0.006, 0.071, 0.066, 0.926)},
}

# Table 3 (\label{gamma_table}) -- Gamma frailty, keyed (setting, method, n)
TABLE3 = {
    (5, 'ODE-Cox', 2000):  {'beta_2': (0.001, 0.072, 0.066, 0.937),
                            'beta_3': (-0.001, 0.069, 0.066, 0.929)},
    (5, 'ODE-Flex', 2000): {'beta_2': (0.006, 0.088, 0.107, 0.958),
                            'beta_3': (0.004, 0.088, 0.107, 0.958)},
    (5, 'Reda', 2000):     {'beta_2': (0.002, 0.053, 0.052, 0.948),
                            'beta_3': (-0.000, 0.053, 0.052, 0.942)},
    (6, 'ODE-AM', 2000):   {'beta_2': (-0.004, 0.070, 0.070, 0.945),
                            'beta_3': (-0.005, 0.070, 0.070, 0.948)},
    (6, 'ODE-Flex', 2000): {'beta_2': (0.005, 0.085, 0.110, 0.968),
                            'beta_3': (0.002, 0.089, 0.108, 0.960)},
    (6, 'Reda', 2000):     {'beta_2': (-0.328, 0.038, 0.042, 0.000),
                            'beta_3': (-0.329, 0.041, 0.042, 0.000)},
    (5, 'ODE-Cox', 4000):  {'beta_2': (0.000, 0.052, 0.048, 0.941),
                            'beta_3': (-0.000, 0.051, 0.048, 0.942)},
    (5, 'ODE-Flex', 4000): {'beta_2': (0.004, 0.065, 0.073, 0.962),
                            'beta_3': (0.007, 0.064, 0.074, 0.955)},
    (5, 'Reda', 4000):     {'beta_2': (0.001, 0.038, 0.037, 0.934),
                            'beta_3': (-0.001, 0.036, 0.036, 0.954)},
    (6, 'ODE-AM', 4000):   {'beta_2': (-0.003, 0.050, 0.051, 0.945),
                            'beta_3': (-0.004, 0.050, 0.051, 0.950)},
    (6, 'ODE-Flex', 4000): {'beta_2': (-0.001, 0.064, 0.086, 0.963),
                            'beta_3': (0.001, 0.063, 0.087, 0.967)},
    (6, 'Reda', 4000):     {'beta_2': (-0.331, 0.029, 0.030, 0.000),
                            'beta_3': (-0.330, 0.028, 0.030, 0.000)},
}

# which local study reproduces which published row
STUDY_TO_PAPER = {
    't1_cox_s1': ('T1', (1, 'ODE-Cox')),
    't1_am_s2': ('T1', (2, 'ODE-AM')),
    't1_lt_s3': ('T1', (3, 'ODE-LT')),
    't2_flex_s1': ('T2', 1),
    't2_flex_s2': ('T2', 2),
    't2_flex_s3': ('T2', 3),
    't2_flex_s4': ('T2', 4),
    't3_cox_s5_n2000': ('T3', (5, 'ODE-Cox', 2000)),
    't3_cox_s5_n4000': ('T3', (5, 'ODE-Cox', 4000)),
    't3_am_s6_n2000': ('T3', (6, 'ODE-AM', 2000)),
    't3_am_s6_n4000': ('T3', (6, 'ODE-AM', 4000)),
    't3_flex_s5_n2000': ('T3', (5, 'ODE-Flex', 2000)),
    't3_flex_s5_n4000': ('T3', (5, 'ODE-Flex', 4000)),
    't3_flex_s6_n2000': ('T3', (6, 'ODE-Flex', 2000)),
    't3_flex_s6_n4000': ('T3', (6, 'ODE-Flex', 4000)),
}


def paper_row(slug):
    """Return ``{'beta_2': (bias, se, ese, cp), ...}`` published for ``slug``."""
    which, key = STUDY_TO_PAPER[slug]
    return {'T1': TABLE1, 'T2': TABLE2, 'T3': TABLE3}[which][key]
