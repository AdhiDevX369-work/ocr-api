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

class InvestigationItem(BaseModel):
    section: Optional[str] = Field(None, description="Section heading (e.g. 'LEUCOCYTES', 'ERYTHROCYTES', 'LIPID PROFILE')")
    investigation: str = Field(..., description="Test parameter name (e.g. 'Haemoglobin', 'W.B.C', 'Total Cholesterol')")
    observed_value: Union[str, int, float] = Field(..., description="Printed result value (e.g. '13.2', 7500, '180')")
    flag: Optional[str] = Field(None, description="Abnormality flag ('H' for High, 'L' for Low, '*' if flagged)")
    unit: Optional[str] = Field(None, description="Measurement unit (e.g. 'g/dL', '/cumm', 'mg/dL', 'mmol/L')")
    reference_interval: Optional[Union[str, int, float]] = Field(None, description="Biological reference interval / normal range")
    is_abnormal: Optional[bool] = Field(False, description="Calculated or flagged indicator if value is out of range")

class SignatureItem(BaseModel):
    signatory_name: Optional[str] = Field(None, description="Name of doctor or lab technician")
    designation: Optional[str] = Field(None, description="Designation (e.g. 'Consultant Haematologist', 'MLT')")

class AdditionalTable(BaseModel):
    table_name: str = Field(..., description="Table title (e.g. 'AVERAGE ESTIMATED GFR BY AGE', 'NCEP ATP III GUIDELINES')")
    headers: List[str] = Field(default_factory=list, description="Column header titles")
    rows: List[List[str]] = Field(default_factory=list, description="Row values")

class MedicalReportExtraction(BaseModel):
    report_title: Optional[str] = Field("Medical Report", description="Title of report (e.g. 'FULL BLOOD COUNT', 'LIPID PROFILE', 'eGFR')")
    patient_info: PatientInfo = Field(default_factory=PatientInfo, description="Patient metadata")
    investigations: List[InvestigationItem] = Field(default_factory=list, description="Extracted lab test parameters")
    additional_tables: Optional[List[AdditionalTable]] = Field(default_factory=list, description="Reference grids / risk charts")
    footnotes: Optional[str] = Field(None, description="Footnotes, methodologies, or instrument notes")
    signatures: Optional[List[SignatureItem]] = Field(default_factory=list, description="Doctor and technician signatures")
    raw_text: Optional[str] = Field(None, description="Complete transcribed text of document")
