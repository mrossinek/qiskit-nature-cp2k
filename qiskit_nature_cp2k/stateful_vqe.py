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
from time import time

from qiskit.algorithms.list_or_dict import ListOrDict
from qiskit.algorithms.minimum_eigensolvers.vqe import VQE, VQEResult
from qiskit.algorithms.optimizers import OptimizerResult
from qiskit.algorithms.observables_evaluator import estimate_observables
from qiskit.algorithms.utils import validate_bounds, validate_initial_point
from qiskit.algorithms.utils.set_batching import _set_default_batchsize
from qiskit.quantum_info.operators.base_operator import BaseOperator

LOGGER = logging.getLogger(__name__)


def depth_filter(param):
    inst, qubits, clbits = param
    return inst.num_qubits == 2


class StatefulVQE(VQE):
    def compute_minimum_eigenvalue(
        self,
        operator: BaseOperator,
        aux_operators: ListOrDict[BaseOperator] | None = None,
    ) -> VQEResult:
        if self.ansatz.num_parameters == 0:
            eigenvalue = estimate_observables(self.estimator, self.ansatz, [operator])[0][0]

            optimizer_result = OptimizerResult()
            optimizer_result.x = []
            optimizer_result.fun = eigenvalue
            optimizer_result.jac = None
            optimizer_result.nfev = 0
            optimizer_result.njev = 0
            optimizer_result.nit = 0

            optimizer_time = 0

            if aux_operators is not None:
                aux_operators_evaluated = estimate_observables(
                    self.estimator,
                    self.ansatz,
                    aux_operators,
                    optimizer_result.x,
                )
            else:
                aux_operators_evaluated = None

            return self._build_vqe_result(
                self.ansatz,
                optimizer_result,
                aux_operators_evaluated,
                optimizer_time,
            )

        self._check_operator_ansatz(operator)

        initial_point = validate_initial_point(self.initial_point, self.ansatz)

        bounds = validate_bounds(self.ansatz)

        start_time = time()

        evaluate_energy = self._get_evaluate_energy(self.ansatz, operator)

        if self.gradient is not None:
            evaluate_gradient = self._get_evaluate_gradient(self.ansatz, operator)
        else:
            evaluate_gradient = None

        # perform optimization
        if callable(self.optimizer):
            optimizer_result = self.optimizer(
                fun=evaluate_energy,
                x0=initial_point,
                jac=evaluate_gradient,
                bounds=bounds,
            )
        else:
            # we always want to submit as many estimations per job as possible for minimal
            # overhead on the hardware
            was_updated = _set_default_batchsize(self.optimizer)

            optimizer_result = self.optimizer.minimize(
                fun=evaluate_energy,
                x0=initial_point,
                jac=evaluate_gradient,
                bounds=bounds,
            )

            # reset to original value
            if was_updated:
                self.optimizer.set_max_evals_grouped(None)

        optimizer_time = time() - start_time

        LOGGER.info(
            "Optimization complete in %s seconds.\nFound optimal point %s",
            optimizer_time,
            optimizer_result.x,
        )

        # stateful aspect to permit warm-starting of the algorithm
        self.initial_point = optimizer_result.x

        if aux_operators is not None:
            aux_operators_evaluated = estimate_observables(
                self.estimator, self.ansatz, aux_operators, optimizer_result.x
            )
        else:
            aux_operators_evaluated = None

        decomposed = self.ansatz.decompose().decompose().decompose()
        LOGGER.info(f"The circuit has the following gates: %s", str(decomposed.count_ops()))
        LOGGER.info(f"The circuit has a depth of %s", str(decomposed.depth()))
        LOGGER.info(f"The circuit has a 2-qubit gate depth of %s", str(decomposed.depth(depth_filter)))
        LOGGER.info(f"The circuit has %s paramters", str(decomposed.num_parameters))

        return self._build_vqe_result(
            self.ansatz, optimizer_result, aux_operators_evaluated, optimizer_time
        )
