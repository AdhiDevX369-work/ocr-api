from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field

class PatientInfo(BaseModel):
    patient_name: Optional[Union[str, int]] = Field(None, description="Full patient name")
    pid_no: Optional[Union[str, int]] = Field(None, description="Patient ID / Registration Number")
    tel_no: Optional[Union[str, int]] = Field(None, description="Contact phone number")
    age: Optional[Union[str, int, float]] = Field(None, description="Age of patient (e.g. '58 Y', '24 Years', 20)")
    sex: Optional[str] = Field(None, description="Gender / Sex (e.g. 'Male', 'Female')")
    reference_dr: Optional[str] = Field(None, description="Referring doctor name")
    sample_collected_at: Optional[str] = Field(None, description="Collecting hospital / laboratory branch")
    collecting_center: Optional[str] = Field(None, description="Collecting center / clinic")
    registered_on: Optional[str] = Field(None, description="Registration date & timestamp")
    collected_on: Optional[str] = Field(None, description="Sample collection date & timestamp")
    reported_on: Optional[str] = Field(None, description="Lab report release date & timestamp")

class ResultItem(BaseModel):
    type: str = Field(..., description="Standardized slug/type of the test (e.g. 'hba1c', 'fbs', 'wbc_count', 'hemoglobin')")
    name: str = Field(..., description="Human readable parameter name (e.g. 'HbA1C', 'FBS', 'WBC Count')")
    value: Union[str, int, float] = Field("", description="Observed parameter value (e.g. '85', '12.0', '')")
    unit: Optional[str] = Field("", description="Measurement unit (e.g. '%', 'mg/dL', 'g/dL')")

# Backward compatibility aliases
class InvestigationItem(BaseModel):
    type: Optional[str] = Field(None)
    name: Optional[str] = Field(None)
    investigation: Optional[str] = Field(None)
    observed_value: Optional[Union[str, int, float]] = Field("")
    value: Optional[Union[str, int, float]] = Field("")
    unit: Optional[str] = Field("")

class SignatureItem(BaseModel):
    signatory_name: Optional[str] = Field(None)
    designation: Optional[str] = Field(None)

class AdditionalTable(BaseModel):
    table_name: str = Field(...)
    headers: List[str] = Field(default_factory=list)
    rows: List[List[str]] = Field(default_factory=list)

class MedicalReportExtraction(BaseModel):
    report_title: Optional[str] = Field("Medical Report", description="Title of report (e.g. 'FULL BLOOD COUNT', 'LIPID PROFILE', 'eGFR')")
    patient_info: PatientInfo = Field(default_factory=PatientInfo, description="Patient metadata")
    results: List[ResultItem] = Field(default_factory=list, description="Standardized list of test results")
    raw_text: Optional[str] = Field(None, description="Complete transcribed text of document")
