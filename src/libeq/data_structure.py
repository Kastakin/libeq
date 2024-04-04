from functools import cached_property
import numpy as np
from pydantic import BaseModel, ConfigDict, computed_field, model_validator
from pydantic_numpy.typing import Np2DArrayInt8, Np1DArrayFp64
import json
from typing import Any, Dict, List

from .parsers.bstac import parse_file


class DistributionOptions(BaseModel):
    initial_log: float | None = None
    final_log: float | None = None
    log_increments: float | None = None

    independent_component: int | None = None


class SolverData(BaseModel):
    """
    Represents the data structure used for solving equations in the libeq library.

    Attributes:
        model_config (ConfigDict): Configuration dictionary for the model.
        components (List[str]): List of component names.
        stoichiometry (Np2DArrayInt8): Stoichiometric matrix.
        solid_stoichiometry (Np2DArrayInt8): Stoichiometric matrix for solid components.
        log_beta (Np1DArrayFp64): Array of logarithmic beta values.
        log_ks (Np1DArrayFp64): Array of logarithmic Ks values.
        charges (Np1DArrayFp64): Array of charges for each component.
        species_charges (Np1DArrayFp64): Array of charges for each species.
        c0 (Np1DArrayFp64 | None): Initial concentrations of components.
        ct (Np1DArrayFp64 | None): Total concentrations of components.
        v0 (float | None): Initial volume.
        v_add (float | Np1DArrayFp64 | None): Additional volume(s).
        num_add (int | None): Number of additional volumes.
        ionic_strength_dependence (bool): Flag indicating if ionic strength dependence is considered.
        ref_ionic_str (float | Np1DArrayFp64): Reference ionic strength value(s).
        z_star (Np1DArrayFp64): Array of z* values.
        p_star (Np1DArrayFp64): Array of p* values.
        dbh_params (Np1DArrayFp64): Array of parameters for the Debye-Hückel equation.

    Methods:
        compute_fields(): Computes additional fields based on the existing data.
        dbh_values(): Returns a dictionary of Debye-Hückel values.
        species_names(): Returns a list of species names.
        nc(): Returns the number of components.
        ns(): Returns the number of species.
        nf(): Returns the number of solid components.
        load_from_bstac(file_path): Loads data from a BSTAC file.
        load_from_json(file_path): Loads data from a JSON file.
    """

    model_config = ConfigDict(extra="forbid")

    distribution_opts: DistributionOptions = DistributionOptions()

    components: List[str]
    stoichiometry: Np2DArrayInt8
    solid_stoichiometry: Np2DArrayInt8 = np.empty((0, 0), dtype=np.int8)
    log_beta: Np1DArrayFp64
    log_ks: Np1DArrayFp64 = np.array([])
    charges: Np1DArrayFp64 = np.array([])
    species_charges: Np1DArrayFp64 = np.array([])
    c0: Np1DArrayFp64 | None = None
    ct: Np1DArrayFp64 | None = None
    v0: float | None = None
    v_add: float | Np1DArrayFp64 | None = None
    num_add: int | None = None
    ionic_strength_dependence: bool = False
    ref_ionic_str: float | Np1DArrayFp64 = 0.0
    z_star: Np1DArrayFp64 = np.array([])
    p_star: Np1DArrayFp64 = np.array([])
    dbh_params: Np1DArrayFp64 = np.array([])

    azast: Np1DArrayFp64 = np.array([])
    bdh: float = 0.0
    cdh: Np1DArrayFp64 = np.array([])
    ddh: Np1DArrayFp64 = np.array([])
    edh: Np1DArrayFp64 = np.array([])
    fib: Np1DArrayFp64 = np.array([])

    @model_validator(mode="after")
    def compute_fields(self) -> Dict[str, Any]:
        self.species_charges = (self.stoichiometry * self.charges[:, np.newaxis]).sum(
            axis=0
        )

        self.ref_ionic_str = np.full(self.stoichiometry.shape[1], 0.1)

        self.z_star = (self.stoichiometry * (self.charges[:, np.newaxis] ** 2)).sum(
            axis=0
        ) - self.species_charges**2

        self.p_star = self.stoichiometry.sum(axis=0) - 1

        self.azast = self.dbh_params[0] * self.z_star
        self.bdh = self.dbh_params[1]
        self.cdh = self.dbh_params[2] * self.p_star + self.dbh_params[3] * self.z_star
        self.ddh = self.dbh_params[4] * self.p_star + self.dbh_params[5] * self.z_star
        self.edh = self.dbh_params[6] * self.p_star + self.dbh_params[7] * self.z_star
        self.fib = np.sqrt(self.ref_ionic_str) / (
            1 + self.dbh_params[1] * np.sqrt(self.ref_ionic_str)
        )

        return self

    @computed_field
    @cached_property
    def dbh_values(self) -> Dict[str, Np1DArrayFp64]:
        return {
            "azast": self.azast,
            "adh": self.dbh_params[0],
            "bdh": self.bdh,
            "cdh": self.cdh,
            "ddh": self.ddh,
            "edh": self.edh,
            "fib": self.fib,
        }

    @computed_field
    @cached_property
    def species_names(self) -> List[str]:
        species_names = ["" for _ in range(self.ns)]
        for i, row in enumerate(self.stoichiometry):
            for j, value in enumerate(row):
                if value < 0:
                    species_names[j] += f"(OH){value if value != -1 else ''}"
                elif value > 0:
                    species_names[j] += (
                        f"({self.components[i]}){value if value != 1 else ''}"
                    )

        return self.components + species_names

    @computed_field
    @cached_property
    def nc(self) -> int:
        return self.stoichiometry.shape[0]

    @computed_field
    @cached_property
    def ns(self) -> int:
        return self.stoichiometry.shape[1]

    @computed_field
    @cached_property
    def nf(self) -> int:
        return self.solid_stoichiometry.shape[0]

    @classmethod
    def load_from_bstac(cls, file_path: str) -> "SolverData":
        data = dict()
        with open(file_path, "r") as file:
            lines = file.readlines()
            parsed_data = parse_file(lines)
            data["stoichiometry"] = np.array(
                [
                    [d[key] for key in d if key.startswith("IX")]
                    for d in parsed_data["species"]
                ]
            ).T
            data["log_beta"] = np.array([d["BLOG"] for d in parsed_data["species"]])
            data["v_add"] = np.array(parsed_data["titrations"][0]["volume"])
            data["c0"] = np.array(
                [
                    d["C0"]
                    for d in parsed_data["titrations"][0]["components_concentrations"]
                ]
            )
            data["ct"] = np.array(
                [
                    d["CTT"]
                    for d in parsed_data["titrations"][0]["components_concentrations"]
                ]
            )
            data["v0"] = parsed_data["titrations"][0]["v_params"][0]

            data["charges"] = np.array(parsed_data.get("charges", []))
            data["components"] = parsed_data["comp_name"]
            data["ionic_strength_dependence"] = parsed_data["ICD"] != 0
            data["dbh_params"] = [
                parsed_data[i] for i in ["AT", "BT", "c0", "c1", "d0", "d1", "e0", "e1"]
            ]
        return cls(**data)

    @classmethod
    def load_from_json(cls, file_path: str) -> "SolverData":
        with open(file_path, "r") as file:
            data = json.load(file)
        return cls(**data)
