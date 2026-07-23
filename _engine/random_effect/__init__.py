"""Python port of the ``random effect`` MATLAB modules.

MATLAB kept each model (Cox, AFT, LTM) in its own folder under
``random effect/``.  Python's package system does not allow a space in a
package name, so the folder is renamed ``random_effect`` while keeping
the per-model subpackages (``cox``, ``aft_rec``, ``ltm``) untouched.
"""
