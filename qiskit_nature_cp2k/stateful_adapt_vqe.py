# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from qiskit import QiskitError
from qiskit.algorithms import estimate_observables
from qiskit.algorithms.minimum_eigensolvers.adapt_vqe import (
    AdaptVQE, AdaptVQEResult, TerminationCriterion)
from qiskit.algorithms.minimum_eigensolvers.vqe import VQEResult
from qiskit.circuit.library import EvolvedOperatorAnsatz

LOGGER = logging.getLogger(__name__)


def depth_filter(param):
    inst, qubits, clbits = param
    return inst.num_qubits == 2


class StatefulAdaptVQE(AdaptVQE):
    """A stateful AdaptVQE variant."""

    def compute_minimum_eigenvalue(
        self, operator, aux_operators=None
    ) -> AdaptVQEResult:
        if not isinstance(self.solver.ansatz, EvolvedOperatorAnsatz) and not isinstance(self._tmp_ansatz, EvolvedOperatorAnsatz):
            raise TypeError(
                "The AdaptVQE ansatz must be of the EvolvedOperatorAnsatz type."
            )

        if self._tmp_ansatz is None:
            # Overwrite the solver's ansatz with the initial state
            self._tmp_ansatz = self.solver.ansatz
            self._excitation_pool = self._tmp_ansatz.operators
            self.solver.ansatz = self._tmp_ansatz.initial_state
            self._theta: list[float] = []
            self._excitation_list = []
            self._global_iteration = 1
            self._prev_op_indices: list[int] = []
            self._history: list[complex] = []
            self._prev_raw_vqe_result: VQEResult | None = None
        else:
            if len(self._excitation_list) == 0:
                self.solver.ansatz = self._tmp_ansatz.initial_state
            else:
                self.solver.ansatz = self._tmp_ansatz
                self.solver.initial_point = self._theta
            self._global_iteration += 1

        raw_vqe_result: VQEResult | None = None
        max_grad: tuple[complex, dict[str, Any] | None] = (0.0, None)
        iteration = 0
        while self.max_iterations is None or iteration < self.max_iterations:
            iteration += 1
            LOGGER.info("--- Iteration #%s ---", str(iteration))
            # compute gradients
            LOGGER.debug("Computing gradients")
            cur_grads = self._compute_gradients(self._theta, operator)
            # pick maximum gradient
            max_grad_index, max_grad = max(
                enumerate(cur_grads), key=lambda item: np.abs(item[1][0])
            )
            LOGGER.info(
                "Found maximum gradient %s at index %s",
                str(np.abs(max_grad[0])),
                str(max_grad_index),
            )
            # log gradients
            if np.abs(max_grad[0]) < self.gradient_threshold:
                if iteration == 1 and self._global_iteration == 1:
                    LOGGER.warning(
                        "All gradients have been evaluated to lie below the convergence threshold "
                        "during the first iteration of the algorithm. Try to either tighten the "
                        "convergence threshold or pick a different ansatz."
                    )
                    raw_vqe_result = self.solver.compute_minimum_eigenvalue(operator)
                    # store this current VQE result for the potential stateful restart later on
                    self._prev_raw_vqe_result = raw_vqe_result
                    self._theta = raw_vqe_result.optimal_point
                    termination_criterion = TerminationCriterion.CONVERGED
                    break
                LOGGER.info(
                    "AdaptVQE terminated successfully with a final maximum gradient: %s",
                    str(np.abs(max_grad[0])),
                )
                termination_criterion = TerminationCriterion.CONVERGED
                break
            # store maximum gradient's index for cycle detection
            self._prev_op_indices.append(max_grad_index)
            # check indices of picked gradients for cycles
            if self._check_cyclicity(self._prev_op_indices):
                LOGGER.info("Alternating sequence found. Finishing.")
                LOGGER.info("Final maximum gradient: %s", str(np.abs(max_grad[0])))
                termination_criterion = TerminationCriterion.CYCLICITY
                break
            # add new excitation to self._ansatz
            LOGGER.info(
                "Adding new operator to the ansatz: %s",
                str(self._excitation_pool[max_grad_index]),
            )
            self._excitation_list.append(self._excitation_pool[max_grad_index])
            self._theta.append(0.0)
            # setting up the ansatz for the VQE iteration
            self._tmp_ansatz.operators = self._excitation_list
            self.solver.ansatz = self._tmp_ansatz
            self.solver.initial_point = self._theta
            # evaluating the eigenvalue with the internal VQE
            self._prev_raw_vqe_result = raw_vqe_result
            raw_vqe_result = self.solver.compute_minimum_eigenvalue(operator)
            self._theta = raw_vqe_result.optimal_point.tolist()
            # checking convergence based on the change in eigenvalue
            if iteration > 1:
                eigenvalue_diff = np.abs(raw_vqe_result.eigenvalue - self._history[-1])
                if eigenvalue_diff < self.eigenvalue_threshold:
                    LOGGER.info(
                        "AdaptVQE terminated successfully with a final change in eigenvalue: %s",
                        str(eigenvalue_diff),
                    )
                    termination_criterion = TerminationCriterion.CONVERGED
                    LOGGER.debug(
                        "Reverting the addition of the last excitation to the ansatz since it "
                        "resulted in a change of the eigenvalue below the configured threshold."
                    )
                    self._excitation_list.pop()
                    self._theta.pop()
                    self._tmp_ansatz.operators = self._excitation_list
                    self.solver.ansatz = self._tmp_ansatz
                    self.solver.initial_point = self._theta
                    raw_vqe_result = self._prev_raw_vqe_result
                    break
            # appending the computed eigenvalue to the tracking history
            self._history.append(raw_vqe_result.eigenvalue)
            LOGGER.info("Current eigenvalue: %s", str(raw_vqe_result.eigenvalue))
        else:
            # reached maximum number of iterations
            self._prev_raw_vqe_result = raw_vqe_result
            termination_criterion = TerminationCriterion.MAXIMUM
            LOGGER.info("Maximum number of iterations reached. Finishing.")
            LOGGER.info("Final maximum gradient: %s", str(np.abs(max_grad[0])))

        if raw_vqe_result is None:
            raw_vqe_result = self._prev_raw_vqe_result
        result = AdaptVQEResult()
        result.combine(raw_vqe_result)
        result.num_iterations = iteration
        result.final_max_gradient = max_grad[0]
        result.termination_criterion = termination_criterion
        result.eigenvalue_history = self._history

        # once finished evaluate auxiliary operators if any
        if aux_operators is not None:
            aux_values = estimate_observables(
                self.solver.estimator,
                self.solver.ansatz,
                aux_operators,
                result.optimal_point,
            )
            result.aux_operators_evaluated = aux_values

        LOGGER.info("The final eigenvalue is: %s", str(result.eigenvalue))
        decomposed = self.solver.ansatz.decompose().decompose().decompose()
        LOGGER.info(f"The current circuit has the following gates: %s", str(decomposed.count_ops()))
        LOGGER.info(f"The current circuit has a depth of %s", str(decomposed.depth()))
        LOGGER.info(f"The current circuit has a 2-qubit gate depth of %s", str(decomposed.depth(depth_filter)))
        LOGGER.info(f"The current circuit has %s paramters", str(decomposed.num_parameters))
        return result
