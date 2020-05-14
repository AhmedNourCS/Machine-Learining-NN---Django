from django import forms

class ApprovalForm(forms.Form):
	firstname=forms.CharField(max_length=15, widget=forms.TextInput(attrs={'placeholder':'Enter Firstname'}))
	lastname=forms.CharField(max_length=15, widget=forms.TextInput(attrs={'placeholder':'Enter Lastname'}))
	Dependents=forms.IntegerField( widget=forms.NumberInput(attrs={'placeholder':'Dependents'}))
	ApplicantIncome=forms.IntegerField(widget=forms.NumberInput(attrs={'placeholder':'Applicant Income'}))
	CoapplicantIncome=forms.IntegerField(widget=forms.NumberInput(attrs={'placeholder':'Co-applicant Income'}))
	LoanAmount=forms.IntegerField(widget=forms.NumberInput(attrs={'placeholder':'Loan Amount'}))
	Loan_Amount_Term=forms.IntegerField(widget=forms.NumberInput(attrs={'placeholder':'LoanAmount Term'}))
	Credit_History=forms.IntegerField(widget=forms.NumberInput(attrs={'placeholder':'Credit History'}))
	Gender=forms.ChoiceField(choices=[('Male', 'Male'),('Female', 'Female')])
	Married=forms.ChoiceField(choices=[('Yes', 'Yes'),('No', 'No')])
	Education=forms.ChoiceField(choices=[('Graduate', 'Graduate'),('Not_Graduate', 'Not_Graduate')])
	Self_Employed=forms.ChoiceField(choices=[('Yes', 'Yes'),('No', 'No')])
	Property_Area=forms.ChoiceField(choices=[('Rural', 'Rural'),('Semiurban', 'Semiurban'),('Urban', 'Urban')])



class HomeForm(forms.Form):
	firstname=forms.CharField(max_length=15, widget=forms.TextInput(attrs={'placeholder':'Enter Firstname'}))
