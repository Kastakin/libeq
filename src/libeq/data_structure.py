import json
from functools import cached_property
from typing import Any, Dict, List, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, computed_field
from pydantic_numpy.typing import (
    Np1DArrayFp64,
    Np2DArrayFp64,
    Np2DArrayInt8,
    Np1DArrayBool,
)

from .utils import NumpyEncoder

from .parsers import parse_BSTAC_file


def _assemble_species_names(components, stoichiometry):
    species_names = ["" for _ in range(stoichiometry.shape[1])]
    for i, row in enumerate(stoichiometry):
        for j, value in enumerate(row):
            if value < 0:
                species_names[j] += f"(OH){np.abs(value) if value != -1 else ''}"
            elif value > 0:
                species_names[j] += f"({components[i]}){value if value != 1 else ''}"

    return species_names


def _validate_or_error(
    field: str, condition: bool, error_message: str, errors: Dict[str, str]
) -> Dict[str, str]:
    try:
        if not condition:
            errors[field] = error_message
    except Exception as _:
        errors[field] = error_message

    return errors


class DistributionParameters(BaseModel):
    c0: Np1DArrayFp64 | None = None
    c0_sigma: Np1DArrayFp64 | None = None

    initial_log: float | None = None
    final_log: float | None = None
    log_increments: float | None = None

    independent_component: int | None = None

    cback: float = 0


class TitrationParameters(BaseModel):
    c0: Np1DArrayFp64 | None = None
    c0_sigma: Np1DArrayFp64 | None = None
    ct: Np1DArrayFp64 | None = None
    ct_sigma: Np1DArrayFp64 | None = None

    c0back: float = 0
    ctback: float = 0


class SimulationTitrationParameters(TitrationParameters):
    v0: float | None = None
    v_increment: float | None = None
    n_add: int | None = None


class PotentiometryTitrationsParameters(TitrationParameters):
    electro_active_compoment: int | None = None
    e0: float | None = None
    e0_sigma: float | None = None
    slope: float | None = None
    v0: float | None = None
    v0_sigma: float | None = None
    v_add: Np1DArrayFp64 | None = None
    emf: Np1DArrayFp64 | None = None
    px_range: List[list[float]] = [[0, 0]]
    ignored: Np1DArrayBool | None = None


class PotentiometryOptions(BaseModel):
    titrations: List[PotentiometryTitrationsParameters] = []
    weights: Literal["constants", "calculated", "given"] = "constants"
    beta_flags: List[int] = []
    conc_flags: List[int] = []
    pot_flags: List[int] = []


class SolverData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    distribution_opts: DistributionParameters = DistributionParameters()
    titration_opts: SimulationTitrationParameters = SimulationTitrationParameters()
    potentiometry_opts: PotentiometryOptions = PotentiometryOptions()

    components: List[str]
    stoichiometry: Np2DArrayInt8
    solid_stoichiometry: Np2DArrayInt8
    log_beta: Np1DArrayFp64
    log_beta_sigma: Np1DArrayFp64 = np.array([])
    log_beta_ref_dbh: Np2DArrayFp64 = np.empty((0, 2))
    log_ks: Np1DArrayFp64 = np.array([])
    log_ks_sigma: Np1DArrayFp64 = np.array([])
    log_ks_ref_dbh: Np2DArrayFp64 = np.empty((0, 2))

    charges: Np1DArrayFp64 = np.array([])

    ionic_strength_dependence: bool = False
    reference_ionic_str_species: Np1DArrayFp64 | float = 0
    reference_ionic_str_solids: Np1DArrayFp64 | float = 0
    dbh_params: Np1DArrayFp64 = np.zeros(8)

    @computed_field
    @cached_property
    def model_ready(self) -> tuple[bool, dict[str, str]]:
        fields_to_check = [
            {
                "field": "Components",
                "condition": len(self.components) > 0,
                "error_message": "At least one component must be provided",
            },
            {
                "field": "Stoichiometry",
                "condition": self.stoichiometry.size > 0,
                "error_message": "Stoichiometry matrix must be provided",
            },
        ]

        errors = {}
        for check in fields_to_check:
            errors = _validate_or_error(**check, errors=errors)

        if errors:
            return False, errors
        else:
            return True, errors

    @computed_field
    @cached_property
    def distribution_ready(self) -> tuple[bool, dict[str, str]]:
        fields_to_check = [
            {
                "field": "Initial pX",
                "condition": (self.distribution_opts.initial_log is not None)
                & (self.distribution_opts.initial_log > 0),
                "error_message": "Initial log must be provided",
            },
            {
                "field": "Final pX",
                "condition": (self.distribution_opts.final_log is not None)
                & (self.distribution_opts.final_log > 0),
                "error_message": "Final log must be provided",
            },
            {
                "field": "pX increments",
                "condition": (self.distribution_opts.log_increments is not None)
                & (self.distribution_opts.log_increments > 0),
                "error_message": "Log increments must be provided",
            },
            {
                "field": "Independent component",
                "condition": self.distribution_opts.independent_component is not None,
            },
            {
                "field": "Independent component concentration",
                "condition": (self.distribution_opts.c0 is not None)
                & (
                    self.distribution_opts.c0[
                        self.distribution_opts.independent_component
                    ]
                    > 0
                ),
                "error_message": "Concentration of independent component must be provided",
            },
        ]

        errors = {}
        for check in fields_to_check:
            errors = _validate_or_error(**check, errors=errors)

        if errors:
            return False, errors
        else:
            return True, errors

    @computed_field
    @cached_property
    def titration_ready(self) -> tuple[bool, dict[str, str]]:
        fields_to_check = [
            {
                "field": "Initial Volume",
                "condition": (self.titration_opts.v0 is not None)
                & (self.titration_opts.v0 > 0),
                "error_message": "Initial volume must be provided",
            },
            {
                "field": "Volume increments",
                "condition": (self.titration_opts.v_increment is not None)
                & (self.titration_opts.v_increment > 0),
                "error_message": "Volume increments must be provided",
            },
            {
                "field": "Number of points",
                "condition": (self.titration_opts.n_add is not None)
                & (self.titration_opts.n_add > 0),
                "error_message": "Number of titration points must be provided",
            },
            {
                "field": "Titrant concentration",
                "condition": (self.titration_opts.ct is not None)
                & (self.titration_opts.ct != 0).any(),
                "error_message": "Titrant concentrations must be provided",
            },
        ]

        errors = {}
        for check in fields_to_check:
            errors = _validate_or_error(**check, errors=errors)

        if errors:
            return False, errors
        else:
            return True, errors

    @computed_field
    @cached_property
    def potentiometry_ready(self) -> tuple[bool, dict[str, str]]:
        fields_to_check = [
            {
                "field": "Initial Volume",
                "condition": np.fromiter(
                    (t.v0 is not None for t in self.potentiometry_opts.titrations), bool
                ).all()
                & np.fromiter(
                    (t.v0 > 0 for t in self.potentiometry_opts.titrations), bool
                ).all(),
                "error_message": "Initial volume must be provided for all titrations",
            },
            {
                "field": "Volume sigma",
                "condition": np.fromiter(
                    (
                        t.v0_sigma is not None
                        for t in self.potentiometry_opts.titrations
                    ),
                    bool,
                ).all()
                & np.fromiter(
                    (t.v0_sigma > 0 for t in self.potentiometry_opts.titrations), bool
                ).all(),
                "error_message": "Volume standard deviation must be provided for all titrations",
            },
            {
                "field": "E0",
                "condition": np.fromiter(
                    (t.e0 is not None for t in self.potentiometry_opts.titrations),
                    bool,
                ).all()
                & np.fromiter(
                    (t.e0 > 0 for t in self.potentiometry_opts.titrations), bool
                ).all(),
                "error_message": "Electrode potential must be provided for all titrations",
            },
            {
                "field": "E Sigma",
                "condition": np.fromiter(
                    (
                        t.e0_sigma is not None
                        for t in self.potentiometry_opts.titrations
                    ),
                    bool,
                ).all()
                & np.fromiter(
                    (t.e0_sigma > 0 for t in self.potentiometry_opts.titrations), bool
                ).all(),
                "error_message": "Electrode standard deviation must be provided for all titrations",
            },
            {
                "field": "Slope",
                "condition": np.fromiter(
                    (t.slope is not None for t in self.potentiometry_opts.titrations),
                    bool,
                ).all()
                & np.fromiter(
                    (t.slope != 0 for t in self.potentiometry_opts.titrations), bool
                ).all(),
                "error_message": "Slope must be provided for all titrations",
            },
            {
                "field": "Titrant concentration",
                "condition": np.fromiter(
                    (t.ct is not None for t in self.potentiometry_opts.titrations),
                    bool,
                ).all()
                & np.fromiter(
                    ((t.ct != 0).any() for t in self.potentiometry_opts.titrations),
                    bool,
                ).all(),
                "error_message": "Titrant concentrations must be provided",
            },
            {
                "field": "Initial concentration",
                "condition": np.fromiter(
                    (t.c0 is not None for t in self.potentiometry_opts.titrations),
                    bool,
                ).all()
                & np.fromiter(
                    ((t.c0 != 0).any() for t in self.potentiometry_opts.titrations),
                    bool,
                ).all(),
                "error_message": "Initial concentrations must be provided",
            },
            {
                "field": "Ignored points",
                "condition": np.fromiter(
                    (t.ignored is not None for t in self.potentiometry_opts.titrations),
                    bool,
                ).all()
                & np.fromiter(
                    (
                        ~(t.ignored == True).all()  # noqa: E712
                        for t in self.potentiometry_opts.titrations
                    ),
                    bool,
                ).all(),
                "error_message": "At least one point must not be ignored",
            },
            {
                "field": "Number of points",
                "condition": np.fromiter(
                    (t.v_add is not None for t in self.potentiometry_opts.titrations),
                    bool,
                ).all()
                & np.fromiter(
                    (t.v_add.size > 0 for t in self.potentiometry_opts.titrations),
                    bool,
                ).all(),
                "error_message": "Titrations need at least one point",
            },
            {
                "field": "Optimization targets",
                "condition": np.bool_(
                    np.array((self.potentiometry_opts.beta_flags))
                ).any(),
                "error_message": "No optimization targets selected",
            },
        ]

        errors = {}
        for check in fields_to_check:
            errors = _validate_or_error(**check, errors=errors)

        if errors:
            return False, errors
        else:
            return True, errors

    @computed_field
    @cached_property
    def species_charges(self) -> Np1DArrayFp64:
        return (self.stoichiometry * self.charges[:, np.newaxis]).sum(axis=0)

    @computed_field
    @cached_property
    def solid_charges(self) -> Np1DArrayFp64:
        return (self.solid_stoichiometry * self.charges[:, np.newaxis]).sum(axis=0)

    @computed_field(repr=False)
    @cached_property
    def z_star_species(self) -> Np1DArrayFp64:
        return (self.stoichiometry * (self.charges[:, np.newaxis] ** 2)).sum(
            axis=0
        ) - self.species_charges**2

    @computed_field(repr=False)
    @cached_property
    def p_star_species(self) -> Np1DArrayFp64:
        return self.stoichiometry.sum(axis=0) - 1

    @computed_field(repr=False)
    @cached_property
    def z_star_solids(self) -> Np1DArrayFp64:
        return (self.solid_stoichiometry * (self.charges[:, np.newaxis] ** 2)).sum(
            axis=0
        ) - self.solid_charges**2

    @computed_field(repr=False)
    @cached_property
    def p_star_solids(self) -> Np1DArrayFp64:
        return self.solid_stoichiometry.sum(axis=0)

    @computed_field
    @cached_property
    def dbh_values(self) -> Dict[str, Np1DArrayFp64]:
        result = dict()
        for phase, iref, per_species_cde, z, p in zip(
            ("species", "solids"),
            (self.reference_ionic_str_species, self.reference_ionic_str_solids),
            (self.log_beta_ref_dbh, self.log_ks_ref_dbh),
            (self.z_star_species, self.z_star_solids),
            (self.p_star_species, self.p_star_solids),
        ):
            dbh_values = dict()
            dbh_values["azast"] = self.dbh_params[0] * z
            dbh_values["adh"] = self.dbh_params[0]
            dbh_values["bdh"] = self.dbh_params[1]
            dbh_values["cdh"] = self.dbh_params[2] * p + self.dbh_params[3] * z
            dbh_values["ddh"] = self.dbh_params[4] * p + self.dbh_params[5] * z
            dbh_values["edh"] = self.dbh_params[6] * p + self.dbh_params[7] * z
            dbh_values["fib"] = np.sqrt(iref) / (1 + self.dbh_params[1] * np.sqrt(iref))

            not_zero_columns = np.where(np.any(per_species_cde != 0, axis=0))[0]
            for i in not_zero_columns:
                dbh_values["cdh"][i] = per_species_cde[0][i]
                dbh_values["ddh"][i] = per_species_cde[1][i]
                dbh_values["edh"][i] = per_species_cde[2][i]

            result[phase] = dbh_values
        return result

    @computed_field
    @cached_property
    def species_names(self) -> List[str]:
        return self.components + _assemble_species_names(
            self.components, self.stoichiometry
        )

    @computed_field
    @cached_property
    def solids_names(self) -> List[str]:
        return _assemble_species_names(self.components, self.solid_stoichiometry)

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
        return self.solid_stoichiometry.shape[1]

    @classmethod
    def load_from_bstac(cls, file_path: str) -> "SolverData":
        data = dict()
        with open(file_path, "r") as file:
            lines = file.readlines()
        parsed_data = parse_BSTAC_file(lines)

        temperature = parsed_data["TEMP"]
        data["stoichiometry"] = np.array(
            [
                [d[key] for key in d if key.startswith("IX")]
                for d in parsed_data["species"]
            ]
        ).T
        data["solid_stoichiometry"] = np.empty(
            (data["stoichiometry"].shape[0], 0), dtype=np.int8
        )
        data["log_beta"] = np.array([d["BLOG"] for d in parsed_data["species"]])

        data["charges"] = np.array(parsed_data.get("charges", []))
        data["components"] = parsed_data["comp_name"]
        data["ionic_strength_dependence"] = parsed_data["ICD"] != 0
        if data["ionic_strength_dependence"]:
            data["reference_ionic_str_species"] = np.array(
                [parsed_data["IREF"] for _ in range(data["stoichiometry"].shape[1])]
            )
            data["reference_ionic_str_solids"] = np.array(
                [
                    parsed_data["IREF"]
                    for _ in range(data["solid_stoichiometry"].shape[1])
                ]
            )
            data["dbh_params"] = [
                parsed_data[i] for i in ["AT", "BT", "c0", "c1", "d0", "d1", "e0", "e1"]
            ]

        titration_options = [
            PotentiometryTitrationsParameters(
                c0=np.array([c["C0"] for c in t["components_concentrations"]]),
                ct=np.array([c["CTT"] for c in t["components_concentrations"]]),
                electro_active_compoment=(
                    t["titration_comp_settings"][1]
                    if t["titration_comp_settings"][1] != 0
                    else len(data["components"]) - 1
                ),
                e0=t["potential_params"][0],
                e0_sigma=t["potential_params"][1],
                slope=(
                    t["potential_params"][4]
                    if t["potential_params"][4] != 0
                    else (temperature + 273.15) / 11.6048 * 2.303
                ),
                v0=t["v_params"][0],
                v0_sigma=t["v_params"][1],
                ignored=np.array(t["ignored"]),
                v_add=np.array(t["volume"]),
                emf=np.array(t["potential"]),
                c0back=t["background_params"][0] if "background_params" in t else 0,
                ctback=t["background_params"][1] if "background_params" in t else 0,
                px_range=[[parsed_data["PHI"], parsed_data["PHF"]]],
            )
            for t in parsed_data["titrations"]
        ]

        weights = parsed_data.get("MODE", 1)
        match weights:
            case 0:
                weights = "calculated"
            case 1:
                weights = "constants"
            case 2:
                weights = "given"
            case _:
                raise ValueError("Invalid MODE value")

        data["potentiometry_opts"] = PotentiometryOptions(
            titrations=titration_options,
            weights=weights,
            beta_flags=[s["KEY"] for s in parsed_data["species"]],
            conc_flags=[],
            pot_flags=[],
        )
        return cls(**data)

    @classmethod
    def load_from_pyes(cls, pyes_data: str | dict) -> "SolverData":
        if isinstance(pyes_data, str):
            with open(pyes_data, "r") as file:
                pyes_data = json.load(file)
        data = dict()
        data["components"] = list(pyes_data["compModel"]["Name"].values())

        data["stoichiometry"] = np.row_stack(
            [
                list(pyes_data["speciesModel"][col].values())
                for col in data["components"]
            ]
        )
        data["log_beta"] = np.array(list(pyes_data["speciesModel"]["LogB"].values()))
        data["log_beta_sigma"] = np.array(
            list(pyes_data["speciesModel"]["Sigma"].values())
        )
        data["log_beta_ref_dbh"] = np.vstack(
            (
                list(pyes_data["speciesModel"]["CGF"].values()),
                list(pyes_data["speciesModel"]["DGF"].values()),
                list(pyes_data["speciesModel"]["EGF"].values()),
            )
        )

        data["solid_stoichiometry"] = np.row_stack(
            [
                list(pyes_data["solidSpeciesModel"][col].values())
                for col in data["components"]
            ]
        )
        data["log_ks"] = np.array(
            list(pyes_data["solidSpeciesModel"]["LogKs"].values())
        )
        data["log_ks_sigma"] = np.array(
            list(pyes_data["solidSpeciesModel"]["Sigma"].values())
        )
        data["log_ks_ref_dbh"] = np.vstack(
            (
                list(pyes_data["solidSpeciesModel"]["CGF"].values()),
                list(pyes_data["solidSpeciesModel"]["DGF"].values()),
                list(pyes_data["solidSpeciesModel"]["EGF"].values()),
            )
        )

        data["charges"] = np.array(list(pyes_data["compModel"]["Charge"].values()))
        data["ionic_strength_dependence"] = pyes_data["imode"] != 0
        data["reference_ionic_str_species"] = np.array(
            list(pyes_data["speciesModel"]["Ref. Ionic Str."].values())
        )
        data["reference_ionic_str_species"] = np.where(
            data["reference_ionic_str_species"] == 0,
            pyes_data["ris"],
            data["reference_ionic_str_species"],
        )

        data["reference_ionic_str_solids"] = np.array(
            list(pyes_data["solidSpeciesModel"]["Ref. Ionic Str."].values())
        )
        data["reference_ionic_str_solids"] = np.where(
            data["reference_ionic_str_solids"] == 0,
            pyes_data["ris"],
            data["reference_ionic_str_solids"],
        )

        data["dbh_params"] = [
            pyes_data[name] for name in ["a", "b", "c0", "c1", "d0", "d1", "e0", "e1"]
        ]

        data["distribution_opts"] = DistributionParameters(
            c0=np.array(list(pyes_data.get("concModel", {}).get("C0", {}).values())),
            c0_sigma=np.array(
                list(pyes_data.get("concModel", {}).get("Sigma C0", {}).values())
            ),
            initial_log=pyes_data.get("initialLog", 0.0),
            final_log=pyes_data.get("finalLog", 0.0),
            log_increments=pyes_data.get("logInc", 0.0),
            independent_component=pyes_data.get("ind_comp", 0),
            cback=pyes_data.get("cback", 0.0),
        )

        data["titration_opts"] = SimulationTitrationParameters(
            c0=np.array(list(pyes_data.get("concModel", {}).get("C0", {}).values())),
            c0_sigma=np.array(
                list(pyes_data.get("concModel", {}).get("Sigma C0", {}).values())
            ),
            ct=np.array(list(pyes_data.get("concModel", {}).get("CT", {}).values())),
            ct_sigma=np.array(
                list(pyes_data.get("concModel", {}).get("Sigma CT", {}).values())
            ),
            v0=pyes_data.get("v0", 0.0),
            v_increment=pyes_data.get("vinc", 0.0),
            n_add=pyes_data.get("nop", 0),
            c0back=pyes_data.get("c0back", 0.0),
            ctback=pyes_data.get("ctback", 0.0),
        )

        potentiometry_data = pyes_data["potentiometry_data"]
        titrations = []
        if potentiometry_data["weightsMode"] == 0:
            weights = "constants"
        elif potentiometry_data["weightsMode"] == 1:
            weights = "calculated"
        elif potentiometry_data["weightsMode"] == 2:
            weights = "given"

        for t in potentiometry_data["titrations"]:
            titrations.append(
                PotentiometryTitrationsParameters(
                    c0=np.array(list(t.get("concView", {}).get("C0", {}).values())),
                    ct=np.array(list(t.get("concView", {}).get("CT", {}).values())),
                    c0_sigma=np.array(
                        list(t.get("concView", {}).get("Sigma C0", {}).values())
                    ),
                    ct_sigma=np.array(
                        list(t.get("concView", {}).get("Sigma CT", {}).values())
                    ),
                    electro_active_compoment=t["electroActiveComponent"],
                    e0=t["e0"],
                    e0_sigma=t["eSigma"],
                    slope=t["slope"],
                    v0=t["initialVolume"],
                    v0_sigma=t["vSigma"],
                    ignored=np.array(
                        list(t.get("titrationView", {}).get("0", {}).values())
                    ),
                    v_add=np.array(
                        list(t.get("titrationView", {}).get("1", {}).values())
                    ),
                    emf=np.array(
                        list(t.get("titrationView", {}).get("2", {}).values())
                    ),
                    c0back=t.get("c0back", 0.0),
                    ctback=t.get("ctback", 0.0),
                    px_range=[px_range for px_range in t["pxRange"]],
                )
            )
        data["potentiometry_opts"] = PotentiometryOptions(
            titrations=titrations,
            weights=weights,
            beta_flags=[int(v) for v in potentiometry_data["beta_refine_flags"]],
            conc_flags=[],
            pot_flags=[],
        )
        data["potentiometry_opts"].conc_flags = [
            "constant" if v == 0 else "calculated" if v == 1 else "given"
            for v in data["potentiometry_opts"].conc_flags
        ]
        return cls(**data)

    def to_pyes(self, format: Literal["dict", "json"] = "dict") -> dict[str, Any] | str:
        if isinstance(self.reference_ionic_str_species, (float, int)):
            species_ref_ionic_str = {
                i: self.reference_ionic_str_species for i in range(self.ns)
            }
        else:
            species_ref_ionic_str = {
                i: ris for i, ris in enumerate(self.reference_ionic_str_species)
            }
        soluble_model = {
            f"{comp_name}": {i: v for i, v in enumerate(row)}
            for comp_name, row in zip(self.components, self.stoichiometry)
        }

        if isinstance(self.reference_ionic_str_solids, (float, int)):
            solid_ref_ionic_str = {
                i: self.reference_ionic_str_solids for i in range(self.nf)
            }
        else:
            solid_ref_ionic_str = {
                i: ris for i, ris in enumerate(self.reference_ionic_str_solids)
            }
        solid_model = {
            f"{comp_name}": {i: v for i, v in enumerate(row)}
            for comp_name, row in zip(self.components, self.solid_stoichiometry)
        }

        if self.potentiometry_ready:
            if self.potentiometry_opts.weights == "constants":
                weights_mode = 0
            elif self.potentiometry_opts.weights == "calculated":
                weights_mode = 1
            elif self.potentiometry_opts.weights == "given":
                weights_mode = 2

            potentiometry_section = {
                "potentiometry_data": {
                    "weightsMode": weights_mode,
                    "beta_refine_flags": [
                        bool(f) for f in self.potentiometry_opts.beta_flags
                    ],
                    "titrations": [
                        {
                            "concView": {
                                "C0": {i: c for i, c in zip(self.components, t.c0)},
                                "CT": {i: c for i, c in zip(self.components, t.ct)},
                                "Sigma C0": {
                                    i: 0 for i, _ in zip(self.components, t.c0)
                                },
                                "Sigma CT": {
                                    i: 0 for i, _ in zip(self.components, t.ct)
                                },
                            },
                            "electroActiveComponent": t.electro_active_compoment,
                            "e0": t.e0,
                            "eSigma": t.e0_sigma,
                            "slope": t.slope,
                            "initialVolume": t.v0,
                            "vSigma": t.v0_sigma,
                            "pxRange": self.potentiometry_opts.px_range,
                            "titrationView": {
                                "0": {i: v for i, v in enumerate(t.ignored)},
                                "1": {i: v for i, v in enumerate(t.v_add)},
                                "2": {i: v for i, v in enumerate(t.emf)},
                                "3": {i: 0 for i, _ in enumerate(t.emf)},
                            },
                        }
                        for t in self.potentiometry_opts.titrations
                    ],
                },
            }
        else:
            potentiometry_section = {}

        if self.distribution_ready:
            distribution_section = {
                "ind_comp": self.distribution_opts.independent_component,
                "initialLog": self.distribution_opts.initial_log,
                "finalLog": self.distribution_opts.final_log,
                "logInc": self.distribution_opts.log_increments,
                "cback": self.distribution_opts.cback,
            }
        else:
            distribution_section = {}

        if self.titration_ready:
            titration_section = {
                "v0": self.titration_opts.v0,
                "initv": self.titration_opts.v0,
                "vinc": self.titration_opts.v_increment,
                "nop": self.titration_opts.n_add,
                "c0back": self.titration_opts.c0back,
                "ctback": self.titration_opts.ctback,
            }
        else:
            titration_section = {}

        if self.titration_ready or self.distribution_ready:
            conc_section = {
                "concModel": {
                    "C0": {i: c0 for i, c0 in enumerate(self.distribution_opts.c0)},
                    "Sigma C0": {
                        i: sigma
                        for i, sigma in enumerate(self.distribution_opts.c0_sigma)
                    },
                    "CT": {i: ct for i, ct in enumerate(self.titration_opts.ct)},
                    "Sigma CT": {
                        i: sigma for i, sigma in enumerate(self.titration_opts.ct_sigma)
                    },
                },
            }
        else:
            conc_section = {}

        if self.log_beta_sigma.size == 0:
            self.log_beta_sigma = np.zeros(self.ns)

        if self.log_ks_sigma.size == 0:
            self.log_ks_sigma = np.zeros(self.nf)

        data = {
            "check": "PyES project file --- DO NOT MODIFY THIS LINE!",
            "nc": self.nc,
            "ns": self.ns,
            "np": self.nf,
            "emode": True,
            "imode": 1 if self.ionic_strength_dependence else 0,
            "ris": 0.0,
            "a": self.dbh_params[0],
            "b": self.dbh_params[1],
            "c0": self.dbh_params[2],
            "c1": self.dbh_params[3],
            "d0": self.dbh_params[4],
            "d1": self.dbh_params[5],
            "e0": self.dbh_params[6],
            "e1": self.dbh_params[7],
            "dmode": 0,
            "compModel": {
                "Name": {i: name for i, name in enumerate(self.components)},
                "Charge": {i: charge for i, charge in enumerate(self.charges)},
            },
            "speciesModel": {
                "Ignored": {i: False for i in range(self.ns)},
                "Name": {
                    i: name for i, name in enumerate(self.species_names[self.nc :])
                },
                "LogB": {i: log_b for i, log_b in enumerate(self.log_beta)},
                "Sigma": {i: sigma for i, sigma in enumerate(self.log_beta_sigma)},
                "Ref. Ionic Str.": species_ref_ionic_str,
                "CGF": {i: c for i, c in enumerate(self.dbh_values["species"]["cdh"])},
                "DGF": {i: d for i, d in enumerate(self.dbh_values["species"]["ddh"])},
                "EGF": {i: e for i, e in enumerate(self.dbh_values["species"]["edh"])},
                **soluble_model,
                "Ref. Comp.": {i: self.components[0] for i in range(self.ns)},
            },
            "solidSpeciesModel": {
                "Ignored": {i: False for i in range(self.nf)},
                "Name": {i: name for i, name in enumerate(self.solids_names)},
                "LogKs": {i: log_ks for i, log_ks in enumerate(self.log_ks)},
                "Sigma": {i: sigma for i, sigma in enumerate(self.log_ks_sigma)},
                "Ref. Ionic Str.": solid_ref_ionic_str,
                "CGF": {i: c for i, c in enumerate(self.dbh_values["solids"]["cdh"])},
                "DGF": {i: d for i, d in enumerate(self.dbh_values["solids"]["ddh"])},
                "EGF": {i: e for i, e in enumerate(self.dbh_values["solids"]["edh"])},
                **solid_model,
                "Ref. Comp.": {i: self.components[0] for i in range(self.nf)},
            },
            **conc_section,
            **titration_section,
            **distribution_section,
            **potentiometry_section,
        }

        return (
            data if format == "dict" else json.dumps(data, indent=4, cls=NumpyEncoder)
        )
