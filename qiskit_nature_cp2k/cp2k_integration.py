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
import socket
from enum import Enum

import numpy as np
from qiskit_nature.second_q.algorithms import GroundStateSolver
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.operators.symmetric_two_body import S1Integrals
from qiskit_nature.second_q.problems import (ElectronicBasis,
                                             ElectronicStructureProblem,
                                             ElectronicStructureResult)
from qiskit_nature.second_q.properties import (AngularMomentum,
                                               ElectronicDensity,
                                               Magnetization, ParticleNumber)
from qiskit_nature.second_q.properties.s_operators import (s_minus_operator,
                                                           s_plus_operator,
                                                           s_x_operator,
                                                           s_y_operator,
                                                           s_z_operator)


from .socket import recv_data, send_data

logger = logging.getLogger(__name__)

class CP2KIntegration:
    # pylint: disable=too-many-instance-attributes
    """TODO."""

    class Messages(Enum):
        """TODO."""

        LENGTH = 12
        # fmt: off
        STATUS     = b"STATUS      "
        ONEBODY    = b"ONEBODY     "
        TWOBODY    = b"TWOBODY     "
        GETDENSITY = b"GETDENSITY  "
        QUIT       = b"QUIT        "
        RECEIVED   = b"RECEIVED    "
        HAVEDATA   = b"HAVEDATA    "
        READY      = b"READY       "
        # fmt: on

    def __init__(self, algo: GroundStateSolver) -> None:
        """TODO."""
        logger.info("\n\n ---------- STARTING CP2K INTEGRATION ----------\n")
        self.algo = algo

        self.socket: socket.socket

        self._nspins: int
        self._nmo_active: int
        self._nelec_active: int
        self._multiplicity: int
        self._h1_a: np.ndarray
        self._h2_aa: np.ndarray
        self._h1_b: np.ndarray | None = None
        self._h2_bb: np.ndarray | None = None
        self._h2_ba: np.ndarray | None = None
        self._overlap_ab: np.ndarray | None = None
        self._e_inactive: float

        self._density_active: ElectronicDensity | None = None
        self._result: ElectronicStructureResult | None = None

    def connect_to_socket(self, host: str, port: int, unix: bool) -> None:
        """Connects to a socket."""
        address: str | tuple[str, int]
        if unix:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            address = host
        else:
            # TODO: double check this
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            address = (host, port)

        logger.info("@CLIENT: connecting to socket")
        while sock.connect_ex(address) != 0:
            continue
        logger.info("@CLIENT: connected")

        self.socket = sock

    def run(self) -> None:
        """Runs the main embedding client."""
        # internal state of the client
        have_data = False

        while True:
            # wait for a message from the server here!
            message = self.socket.recv(self.Messages.LENGTH.value)
            logger.info("@CLIENT: Message from server: %s", message)

            if message == self.Messages.STATUS.value:
                if have_data:
                    self.socket.send(self.Messages.HAVEDATA.value)
                else:
                    self.socket.send(self.Messages.READY.value)

            elif message == self.Messages.TWOBODY.value:
                self.process_message_twobody()

            elif message == self.Messages.ONEBODY.value:
                self.process_message_onebody()

                problem = self.construct_problem()

                # at this point I have all the info to run qiskit
                self._result = self.solve(problem)
                self._density_active = self._result.electronic_density
                # change internal status to have data
                have_data = True
                # signal CP2K that I have data
                self.socket.send(self.Messages.HAVEDATA.value)

            elif message == self.Messages.GETDENSITY.value:
                # cp2k asks for the data
                send_data(self.socket, np.float64(self._result.computed_energies[0]))
                if self._nspins == 1:
                    density = self._result.electronic_density.trace_spin()["+-"]
                    send_data(self.socket, density)

                else:
                    density_a = self._result.electronic_density.alpha["+-"]
                    density_b = self._result.electronic_density.beta["+-"]
                    try:
                        magnetization = self._result.magnetization[0]
                    except:
                        magnetization = self._result.magnetization
                    if np.isclose(magnetization, -1):
                        logger.info("magnetization is -1, swapping densities")
                        density_a, density_b = density_b, density_a

                    send_data(self.socket, density_a)
                    send_data(self.socket, density_b)

                # change internal state to no data no more
                have_data = False

            elif message == self.Messages.QUIT.value:
                break

            else:
                print("error handling")

    def construct_problem(self) -> ElectronicStructureProblem:
        """Constructs a Qiskit Nature problem specification from the provided raw data."""
        hamil = ElectronicEnergy.from_raw_integrals(
            self._h1_a,
            S1Integrals(self._h2_aa),
            self._h1_b,
            S1Integrals(self._h2_bb) if self._h2_bb is not None else None,
            S1Integrals(self._h2_ba) if self._h2_ba is not None else None,
            auto_index_order=False,
        )
        hamil.nuclear_repulsion_energy = self._e_inactive
        problem = ElectronicStructureProblem(hamil)
        problem.basis = ElectronicBasis.MO
        problem.properties.angular_momentum = AngularMomentum(self._nmo_active, self._overlap_ab)
        problem.properties.magnetization = Magnetization(self._nmo_active)
        problem.properties.particle_number = ParticleNumber(self._nmo_active)

        problem.num_spatial_orbitals = self._nmo_active
        nalpha = (self._nelec_active + self._multiplicity - 1) // 2
        nbeta = self._nelec_active - nalpha
        problem.num_particles = (nalpha, nbeta)
        occ_a = np.zeros(self._nmo_active)
        occ_a[0:nalpha] = 1.0
        occ_b = np.zeros(self._nmo_active)
        occ_b[0:nbeta] = 1.0
        problem.orbital_occupations = occ_a
        problem.orbital_occupations_b = occ_b

        if self._density_active is None:
            # initialize some stuff
            self._density_active = ElectronicDensity.from_orbital_occupation(
                problem.orbital_occupations,
                problem.orbital_occupations_b,
                include_rdm2=False,
            )

        problem.properties.electronic_density = self._density_active

        return problem

    def process_message_twobody(self) -> None:
        """Processes the TWOBODY message data."""
        # info about the system
        self._nspins = recv_data(self.socket, np.int32(0))
        logger.info("nspins = %s", self._nspins)
        self._nmo_active = recv_data(self.socket, np.int32(0))
        logger.info("nmo_active = %s", self._nmo_active)
        self._nelec_active = recv_data(self.socket, np.int32(0))
        logger.info("nelec_active = %s", self._nelec_active)
        self._multiplicity = recv_data(self.socket, np.int32(0))
        logger.info("multiplicity = %s", self._multiplicity)
        # two-body integrals
        self._h2_aa = np.zeros(
            (self._nmo_active, self._nmo_active, self._nmo_active, self._nmo_active),
            float,
        )
        self._h2_aa = recv_data(self.socket, self._h2_aa)
        if self._nspins == 2:
            _h2_ab = np.zeros(
                (
                    self._nmo_active,
                    self._nmo_active,
                    self._nmo_active,
                    self._nmo_active,
                ),
                float,
            )
            _h2_ab = recv_data(self.socket, _h2_ab)
            # I think that we are already reading the transpose
            # because the combined index ijkl in qiskit is defined
            # in a symmetrically opposite way than in cp2k
            # self._h2_ba = _h2_ab.transpose()
            self._h2_ba = _h2_ab
            self._h2_bb = np.zeros(
                (
                    self._nmo_active,
                    self._nmo_active,
                    self._nmo_active,
                    self._nmo_active,
                ),
                float,
            )
            self._h2_bb = recv_data(self.socket, self._h2_bb)
            # when dealing with unrestricted spin integrals, we also need to obtain the alpha-beta
            # overlap information
            self._overlap_ab = np.zeros((self._nmo_active, self._nmo_active), float)
            self._overlap_ab = recv_data(self.socket, self._overlap_ab)
            logger.info("Dumping the AB overlap matrix: \n%s", self._overlap_ab)

        # signal CP2K that I have data
        self.socket.send(self.Messages.RECEIVED.value)

    def process_message_onebody(self) -> None:
        """Processes the ONEBODY message data."""
        self._e_inactive = recv_data(self.socket, np.float64(0.0))
        logger.info("e_inactive = %s", self._e_inactive)

        self._h1_a = np.zeros((self._nmo_active, self._nmo_active), float)
        self._h1_a = recv_data(self.socket, self._h1_a)

        if self._nspins == 2:
            self._h1_b = np.zeros((self._nmo_active, self._nmo_active), float)
            self._h1_b = recv_data(self.socket, self._h1_b)

        logger.info("h1_a = \n%s", self._h1_a)
        if self._nspins == 2:
            logger.info("h1_b = \n%s", self._h1_b)

    def solve(self, problem: ElectronicStructureProblem) -> ElectronicStructureResult:
        """Solves the provides problem using Qiskit Nature."""

        logger.info("\n\n ------------------  SOLVE  -----------------\n")
        logger.info(problem.num_spatial_orbitals)
        logger.info(problem.num_particles)
        logger.info(problem.orbital_occupations)
        logger.info(problem.orbital_occupations_b)

        result = self.algo.solve(problem)

        logger.info("Active space result: \n\n%s\n", result)

        if self._nspins == 1:
            density = result.electronic_density.trace_spin()["+-"]
            logger.info("Active space density: \n\n%s\n", density)
        else:
            density_a = result.electronic_density.alpha["+-"]
            occ_a = np.linalg.eigvals(density_a)
            logger.info("Active space alpha density: \n\n%s\n", density_a)
            density_b = result.electronic_density.beta["+-"]
            occ_b = np.linalg.eigvals(density_b)
            logger.info("Active space beta density: \n\n%s\n", density_b)
            logger.info("Occupation numbers:\n%s\n", np.asarray([occ_a, occ_b]))

        return result
