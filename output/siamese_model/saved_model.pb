ίχ
¦ό
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
>
Maximum
x"T
y"T
z"T"
Ttype:
2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
-
Sqrt
x"T
y"T"
Ttype:

2
7
Square
x"T
y"T"
Ttype:
2	
Α
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.8.02v2.8.0-0-g3f878cff5b68
z
dense_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_13/kernel
s
#dense_13/kernel/Read/ReadVariableOpReadVariableOpdense_13/kernel*
_output_shapes

:*
dtype0
r
dense_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_13/bias
k
!dense_13/bias/Read/ReadVariableOpReadVariableOpdense_13/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0

conv2d_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameconv2d_12/kernel
}
$conv2d_12/kernel/Read/ReadVariableOpReadVariableOpconv2d_12/kernel*&
_output_shapes
:@*
dtype0
t
conv2d_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_12/bias
m
"conv2d_12/bias/Read/ReadVariableOpReadVariableOpconv2d_12/bias*
_output_shapes
:@*
dtype0

conv2d_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameconv2d_13/kernel
}
$conv2d_13/kernel/Read/ReadVariableOpReadVariableOpconv2d_13/kernel*&
_output_shapes
:@@*
dtype0
t
conv2d_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_13/bias
m
"conv2d_13/bias/Read/ReadVariableOpReadVariableOpconv2d_13/bias*
_output_shapes
:@*
dtype0
z
dense_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@0* 
shared_namedense_12/kernel
s
#dense_12/kernel/Read/ReadVariableOpReadVariableOpdense_12/kernel*
_output_shapes

:@0*
dtype0
r
dense_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*
shared_namedense_12/bias
k
!dense_12/bias/Read/ReadVariableOpReadVariableOpdense_12/bias*
_output_shapes
:0*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

Adam/dense_13/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_13/kernel/m

*Adam/dense_13/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_13/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_13/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_13/bias/m
y
(Adam/dense_13/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_13/bias/m*
_output_shapes
:*
dtype0

Adam/conv2d_12/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/conv2d_12/kernel/m

+Adam/conv2d_12/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_12/kernel/m*&
_output_shapes
:@*
dtype0

Adam/conv2d_12/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_12/bias/m
{
)Adam/conv2d_12/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_12/bias/m*
_output_shapes
:@*
dtype0

Adam/conv2d_13/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv2d_13/kernel/m

+Adam/conv2d_13/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_13/kernel/m*&
_output_shapes
:@@*
dtype0

Adam/conv2d_13/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_13/bias/m
{
)Adam/conv2d_13/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_13/bias/m*
_output_shapes
:@*
dtype0

Adam/dense_12/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@0*'
shared_nameAdam/dense_12/kernel/m

*Adam/dense_12/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_12/kernel/m*
_output_shapes

:@0*
dtype0

Adam/dense_12/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*%
shared_nameAdam/dense_12/bias/m
y
(Adam/dense_12/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_12/bias/m*
_output_shapes
:0*
dtype0

Adam/dense_13/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_13/kernel/v

*Adam/dense_13/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_13/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_13/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_13/bias/v
y
(Adam/dense_13/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_13/bias/v*
_output_shapes
:*
dtype0

Adam/conv2d_12/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/conv2d_12/kernel/v

+Adam/conv2d_12/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_12/kernel/v*&
_output_shapes
:@*
dtype0

Adam/conv2d_12/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_12/bias/v
{
)Adam/conv2d_12/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_12/bias/v*
_output_shapes
:@*
dtype0

Adam/conv2d_13/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv2d_13/kernel/v

+Adam/conv2d_13/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_13/kernel/v*&
_output_shapes
:@@*
dtype0

Adam/conv2d_13/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_13/bias/v
{
)Adam/conv2d_13/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_13/bias/v*
_output_shapes
:@*
dtype0

Adam/dense_12/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@0*'
shared_nameAdam/dense_12/kernel/v

*Adam/dense_12/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_12/kernel/v*
_output_shapes

:@0*
dtype0

Adam/dense_12/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*%
shared_nameAdam/dense_12/bias/v
y
(Adam/dense_12/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_12/bias/v*
_output_shapes
:0*
dtype0

NoOpNoOp
₯R
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ΰQ
valueΦQBΣQ BΜQ
Α
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer_with_weights-1
layer-4
	optimizer
	variables
trainable_variables
	regularization_losses

	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
* 
Σ
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer-6
layer-7
layer_with_weights-2
layer-8
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*

	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses* 
¦

$kernel
%bias
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses*
δ
,iter

-beta_1

.beta_2
	/decay
0learning_rate$m±%m²1m³2m΄3m΅4mΆ5m·6mΈ$vΉ%vΊ1v»2vΌ3v½4vΎ5vΏ6vΐ*
<
10
21
32
43
54
65
$6
%7*
<
10
21
32
43
54
65
$6
%7*
* 
°
7non_trainable_variables

8layers
9metrics
:layer_regularization_losses
;layer_metrics
	variables
trainable_variables
	regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

<serving_default* 
* 
¦

1kernel
2bias
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses*

C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses* 
₯
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M_random_generator
N__call__
*O&call_and_return_all_conditional_losses* 
¦

3kernel
4bias
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses*

V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses* 
₯
\	variables
]trainable_variables
^regularization_losses
_	keras_api
`_random_generator
a__call__
*b&call_and_return_all_conditional_losses* 

c	variables
dtrainable_variables
eregularization_losses
f	keras_api
g__call__
*h&call_and_return_all_conditional_losses* 
¦

5kernel
6bias
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses*
.
10
21
32
43
54
65*
.
10
21
32
43
54
65*
* 

onon_trainable_variables

players
qmetrics
rlayer_regularization_losses
slayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

tnon_trainable_variables

ulayers
vmetrics
wlayer_regularization_losses
xlayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses* 
* 
* 
_Y
VARIABLE_VALUEdense_13/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_13/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

$0
%1*

$0
%1*
* 

ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses*
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEconv2d_12/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEconv2d_12/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEconv2d_13/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEconv2d_13/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_12/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_12/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
* 
'
0
1
2
3
4*

~0
1*
* 
* 
* 

10
21*

10
21*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses* 
* 
* 
* 

30
41*

30
41*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
\	variables
]trainable_variables
^regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 

non_trainable_variables
layers
 metrics
 ‘layer_regularization_losses
’layer_metrics
c	variables
dtrainable_variables
eregularization_losses
g__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses* 
* 
* 

50
61*

50
61*
* 

£non_trainable_variables
€layers
₯metrics
 ¦layer_regularization_losses
§layer_metrics
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses*
* 
* 
* 
C
0
1
2
3
4
5
6
7
8*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<

¨total

©count
ͺ	variables
«	keras_api*
M

¬total

­count
?
_fn_kwargs
―	variables
°	keras_api*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

¨0
©1*

ͺ	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

¬0
­1*

―	variables*
|
VARIABLE_VALUEAdam/dense_13/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_13/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_12/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_12/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_13/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_13/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/dense_12/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/dense_12/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_13/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_13/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_12/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_12/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_13/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_13/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/dense_12/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/dense_12/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

serving_default_input_19Placeholder*/
_output_shapes
:?????????dd*
dtype0*$
shape:?????????dd

serving_default_input_20Placeholder*/
_output_shapes
:?????????dd*
dtype0*$
shape:?????????dd
α
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_19serving_default_input_20conv2d_12/kernelconv2d_12/biasconv2d_13/kernelconv2d_13/biasdense_12/kerneldense_12/biasdense_13/kerneldense_13/bias*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference_signature_wrapper_23269
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
°
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_13/kernel/Read/ReadVariableOp!dense_13/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp$conv2d_12/kernel/Read/ReadVariableOp"conv2d_12/bias/Read/ReadVariableOp$conv2d_13/kernel/Read/ReadVariableOp"conv2d_13/bias/Read/ReadVariableOp#dense_12/kernel/Read/ReadVariableOp!dense_12/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/dense_13/kernel/m/Read/ReadVariableOp(Adam/dense_13/bias/m/Read/ReadVariableOp+Adam/conv2d_12/kernel/m/Read/ReadVariableOp)Adam/conv2d_12/bias/m/Read/ReadVariableOp+Adam/conv2d_13/kernel/m/Read/ReadVariableOp)Adam/conv2d_13/bias/m/Read/ReadVariableOp*Adam/dense_12/kernel/m/Read/ReadVariableOp(Adam/dense_12/bias/m/Read/ReadVariableOp*Adam/dense_13/kernel/v/Read/ReadVariableOp(Adam/dense_13/bias/v/Read/ReadVariableOp+Adam/conv2d_12/kernel/v/Read/ReadVariableOp)Adam/conv2d_12/bias/v/Read/ReadVariableOp+Adam/conv2d_13/kernel/v/Read/ReadVariableOp)Adam/conv2d_13/bias/v/Read/ReadVariableOp*Adam/dense_12/kernel/v/Read/ReadVariableOp(Adam/dense_12/bias/v/Read/ReadVariableOpConst*.
Tin'
%2#	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *'
f"R 
__inference__traced_save_23704

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_13/kerneldense_13/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateconv2d_12/kernelconv2d_12/biasconv2d_13/kernelconv2d_13/biasdense_12/kerneldense_12/biastotalcounttotal_1count_1Adam/dense_13/kernel/mAdam/dense_13/bias/mAdam/conv2d_12/kernel/mAdam/conv2d_12/bias/mAdam/conv2d_13/kernel/mAdam/conv2d_13/bias/mAdam/dense_12/kernel/mAdam/dense_12/bias/mAdam/dense_13/kernel/vAdam/dense_13/bias/vAdam/conv2d_12/kernel/vAdam/conv2d_12/bias/vAdam/conv2d_13/kernel/vAdam/conv2d_13/bias/vAdam/dense_12/kernel/vAdam/dense_12/bias/v*-
Tin&
$2"*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__traced_restore_23813Λέ	
?
λ
C__inference_model_13_layer_call_and_return_conditional_losses_22818

inputs
inputs_1(
model_12_22764:@
model_12_22766:@(
model_12_22768:@@
model_12_22770:@ 
model_12_22772:@0
model_12_22774:0 
dense_13_22812:
dense_13_22814:
identity’ dense_13/StatefulPartitionedCall’ model_12/StatefulPartitionedCall’"model_12/StatefulPartitionedCall_1΅
 model_12/StatefulPartitionedCallStatefulPartitionedCallinputsmodel_12_22764model_12_22766model_12_22768model_12_22770model_12_22772model_12_22774*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????0*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_model_12_layer_call_and_return_conditional_losses_22541Ή
"model_12/StatefulPartitionedCall_1StatefulPartitionedCallinputs_1model_12_22764model_12_22766model_12_22768model_12_22770model_12_22772model_12_22774*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????0*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_model_12_layer_call_and_return_conditional_losses_22541
lambda_6/PartitionedCallPartitionedCall)model_12/StatefulPartitionedCall:output:0+model_12/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lambda_6_layer_call_and_return_conditional_losses_22798
 dense_13/StatefulPartitionedCallStatefulPartitionedCall!lambda_6/PartitionedCall:output:0dense_13_22812dense_13_22814*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_13_layer_call_and_return_conditional_losses_22811x
IdentityIdentity)dense_13/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????±
NoOpNoOp!^dense_13/StatefulPartitionedCall!^model_12/StatefulPartitionedCall#^model_12/StatefulPartitionedCall_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:?????????dd:?????????dd: : : : : : : : 2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 model_12/StatefulPartitionedCall model_12/StatefulPartitionedCall2H
"model_12/StatefulPartitionedCall_1"model_12/StatefulPartitionedCall_1:W S
/
_output_shapes
:?????????dd
 
_user_specified_nameinputs:WS
/
_output_shapes
:?????????dd
 
_user_specified_nameinputs


o
C__inference_lambda_6_layer_call_and_return_conditional_losses_23403
inputs_0
inputs_1
identityP
subSubinputs_0inputs_1*
T0*'
_output_shapes
:?????????0K
SquareSquaresub:z:0*
T0*'
_output_shapes
:?????????0W
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :y
SumSum
Square:y:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(N
	Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΏΦ3f
MaximumMaximumSum:output:0Maximum/y:output:0*
T0*'
_output_shapes
:?????????J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    c
	Maximum_1MaximumMaximum:z:0Const:output:0*
T0*'
_output_shapes
:?????????M
SqrtSqrtMaximum_1:z:0*
T0*'
_output_shapes
:?????????P
IdentityIdentitySqrt:y:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????0:?????????0:Q M
'
_output_shapes
:?????????0
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????0
"
_user_specified_name
inputs/1
Γ^
έ	
C__inference_model_13_layer_call_and_return_conditional_losses_23150
inputs_0
inputs_1K
1model_12_conv2d_12_conv2d_readvariableop_resource:@@
2model_12_conv2d_12_biasadd_readvariableop_resource:@K
1model_12_conv2d_13_conv2d_readvariableop_resource:@@@
2model_12_conv2d_13_biasadd_readvariableop_resource:@B
0model_12_dense_12_matmul_readvariableop_resource:@0?
1model_12_dense_12_biasadd_readvariableop_resource:09
'dense_13_matmul_readvariableop_resource:6
(dense_13_biasadd_readvariableop_resource:
identity’dense_13/BiasAdd/ReadVariableOp’dense_13/MatMul/ReadVariableOp’)model_12/conv2d_12/BiasAdd/ReadVariableOp’+model_12/conv2d_12/BiasAdd_1/ReadVariableOp’(model_12/conv2d_12/Conv2D/ReadVariableOp’*model_12/conv2d_12/Conv2D_1/ReadVariableOp’)model_12/conv2d_13/BiasAdd/ReadVariableOp’+model_12/conv2d_13/BiasAdd_1/ReadVariableOp’(model_12/conv2d_13/Conv2D/ReadVariableOp’*model_12/conv2d_13/Conv2D_1/ReadVariableOp’(model_12/dense_12/BiasAdd/ReadVariableOp’*model_12/dense_12/BiasAdd_1/ReadVariableOp’'model_12/dense_12/MatMul/ReadVariableOp’)model_12/dense_12/MatMul_1/ReadVariableOp’
(model_12/conv2d_12/Conv2D/ReadVariableOpReadVariableOp1model_12_conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Α
model_12/conv2d_12/Conv2DConv2Dinputs_00model_12/conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????dd@*
paddingSAME*
strides

)model_12/conv2d_12/BiasAdd/ReadVariableOpReadVariableOp2model_12_conv2d_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ά
model_12/conv2d_12/BiasAddBiasAdd"model_12/conv2d_12/Conv2D:output:01model_12/conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????dd@~
model_12/conv2d_12/ReluRelu#model_12/conv2d_12/BiasAdd:output:0*
T0*/
_output_shapes
:?????????dd@ΐ
!model_12/max_pooling2d_12/MaxPoolMaxPool%model_12/conv2d_12/Relu:activations:0*/
_output_shapes
:?????????22@*
ksize
*
paddingVALID*
strides

model_12/dropout_12/IdentityIdentity*model_12/max_pooling2d_12/MaxPool:output:0*
T0*/
_output_shapes
:?????????22@’
(model_12/conv2d_13/Conv2D/ReadVariableOpReadVariableOp1model_12_conv2d_13_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0ή
model_12/conv2d_13/Conv2DConv2D%model_12/dropout_12/Identity:output:00model_12/conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????22@*
paddingSAME*
strides

)model_12/conv2d_13/BiasAdd/ReadVariableOpReadVariableOp2model_12_conv2d_13_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ά
model_12/conv2d_13/BiasAddBiasAdd"model_12/conv2d_13/Conv2D:output:01model_12/conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????22@~
model_12/conv2d_13/ReluRelu#model_12/conv2d_13/BiasAdd:output:0*
T0*/
_output_shapes
:?????????22@ΐ
!model_12/max_pooling2d_13/MaxPoolMaxPool%model_12/conv2d_13/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides

model_12/dropout_13/IdentityIdentity*model_12/max_pooling2d_13/MaxPool:output:0*
T0*/
_output_shapes
:?????????@
:model_12/global_average_pooling2d_6/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      Ξ
(model_12/global_average_pooling2d_6/MeanMean%model_12/dropout_13/Identity:output:0Cmodel_12/global_average_pooling2d_6/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????@
'model_12/dense_12/MatMul/ReadVariableOpReadVariableOp0model_12_dense_12_matmul_readvariableop_resource*
_output_shapes

:@0*
dtype0Έ
model_12/dense_12/MatMulMatMul1model_12/global_average_pooling2d_6/Mean:output:0/model_12/dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????0
(model_12/dense_12/BiasAdd/ReadVariableOpReadVariableOp1model_12_dense_12_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0¬
model_12/dense_12/BiasAddBiasAdd"model_12/dense_12/MatMul:product:00model_12/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????0€
*model_12/conv2d_12/Conv2D_1/ReadVariableOpReadVariableOp1model_12_conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Ε
model_12/conv2d_12/Conv2D_1Conv2Dinputs_12model_12/conv2d_12/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????dd@*
paddingSAME*
strides

+model_12/conv2d_12/BiasAdd_1/ReadVariableOpReadVariableOp2model_12_conv2d_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ό
model_12/conv2d_12/BiasAdd_1BiasAdd$model_12/conv2d_12/Conv2D_1:output:03model_12/conv2d_12/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????dd@
model_12/conv2d_12/Relu_1Relu%model_12/conv2d_12/BiasAdd_1:output:0*
T0*/
_output_shapes
:?????????dd@Δ
#model_12/max_pooling2d_12/MaxPool_1MaxPool'model_12/conv2d_12/Relu_1:activations:0*/
_output_shapes
:?????????22@*
ksize
*
paddingVALID*
strides

model_12/dropout_12/Identity_1Identity,model_12/max_pooling2d_12/MaxPool_1:output:0*
T0*/
_output_shapes
:?????????22@€
*model_12/conv2d_13/Conv2D_1/ReadVariableOpReadVariableOp1model_12_conv2d_13_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0δ
model_12/conv2d_13/Conv2D_1Conv2D'model_12/dropout_12/Identity_1:output:02model_12/conv2d_13/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????22@*
paddingSAME*
strides

+model_12/conv2d_13/BiasAdd_1/ReadVariableOpReadVariableOp2model_12_conv2d_13_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ό
model_12/conv2d_13/BiasAdd_1BiasAdd$model_12/conv2d_13/Conv2D_1:output:03model_12/conv2d_13/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????22@
model_12/conv2d_13/Relu_1Relu%model_12/conv2d_13/BiasAdd_1:output:0*
T0*/
_output_shapes
:?????????22@Δ
#model_12/max_pooling2d_13/MaxPool_1MaxPool'model_12/conv2d_13/Relu_1:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides

model_12/dropout_13/Identity_1Identity,model_12/max_pooling2d_13/MaxPool_1:output:0*
T0*/
_output_shapes
:?????????@
<model_12/global_average_pooling2d_6/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      Τ
*model_12/global_average_pooling2d_6/Mean_1Mean'model_12/dropout_13/Identity_1:output:0Emodel_12/global_average_pooling2d_6/Mean_1/reduction_indices:output:0*
T0*'
_output_shapes
:?????????@
)model_12/dense_12/MatMul_1/ReadVariableOpReadVariableOp0model_12_dense_12_matmul_readvariableop_resource*
_output_shapes

:@0*
dtype0Ύ
model_12/dense_12/MatMul_1MatMul3model_12/global_average_pooling2d_6/Mean_1:output:01model_12/dense_12/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????0
*model_12/dense_12/BiasAdd_1/ReadVariableOpReadVariableOp1model_12_dense_12_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0²
model_12/dense_12/BiasAdd_1BiasAdd$model_12/dense_12/MatMul_1:product:02model_12/dense_12/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????0
lambda_6/subSub"model_12/dense_12/BiasAdd:output:0$model_12/dense_12/BiasAdd_1:output:0*
T0*'
_output_shapes
:?????????0]
lambda_6/SquareSquarelambda_6/sub:z:0*
T0*'
_output_shapes
:?????????0`
lambda_6/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :
lambda_6/SumSumlambda_6/Square:y:0'lambda_6/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(W
lambda_6/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΏΦ3
lambda_6/MaximumMaximumlambda_6/Sum:output:0lambda_6/Maximum/y:output:0*
T0*'
_output_shapes
:?????????S
lambda_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ~
lambda_6/Maximum_1Maximumlambda_6/Maximum:z:0lambda_6/Const:output:0*
T0*'
_output_shapes
:?????????_
lambda_6/SqrtSqrtlambda_6/Maximum_1:z:0*
T0*'
_output_shapes
:?????????
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_13/MatMulMatMullambda_6/Sqrt:y:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????h
dense_13/SigmoidSigmoiddense_13/BiasAdd:output:0*
T0*'
_output_shapes
:?????????c
IdentityIdentitydense_13/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????
NoOpNoOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp*^model_12/conv2d_12/BiasAdd/ReadVariableOp,^model_12/conv2d_12/BiasAdd_1/ReadVariableOp)^model_12/conv2d_12/Conv2D/ReadVariableOp+^model_12/conv2d_12/Conv2D_1/ReadVariableOp*^model_12/conv2d_13/BiasAdd/ReadVariableOp,^model_12/conv2d_13/BiasAdd_1/ReadVariableOp)^model_12/conv2d_13/Conv2D/ReadVariableOp+^model_12/conv2d_13/Conv2D_1/ReadVariableOp)^model_12/dense_12/BiasAdd/ReadVariableOp+^model_12/dense_12/BiasAdd_1/ReadVariableOp(^model_12/dense_12/MatMul/ReadVariableOp*^model_12/dense_12/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:?????????dd:?????????dd: : : : : : : : 2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2V
)model_12/conv2d_12/BiasAdd/ReadVariableOp)model_12/conv2d_12/BiasAdd/ReadVariableOp2Z
+model_12/conv2d_12/BiasAdd_1/ReadVariableOp+model_12/conv2d_12/BiasAdd_1/ReadVariableOp2T
(model_12/conv2d_12/Conv2D/ReadVariableOp(model_12/conv2d_12/Conv2D/ReadVariableOp2X
*model_12/conv2d_12/Conv2D_1/ReadVariableOp*model_12/conv2d_12/Conv2D_1/ReadVariableOp2V
)model_12/conv2d_13/BiasAdd/ReadVariableOp)model_12/conv2d_13/BiasAdd/ReadVariableOp2Z
+model_12/conv2d_13/BiasAdd_1/ReadVariableOp+model_12/conv2d_13/BiasAdd_1/ReadVariableOp2T
(model_12/conv2d_13/Conv2D/ReadVariableOp(model_12/conv2d_13/Conv2D/ReadVariableOp2X
*model_12/conv2d_13/Conv2D_1/ReadVariableOp*model_12/conv2d_13/Conv2D_1/ReadVariableOp2T
(model_12/dense_12/BiasAdd/ReadVariableOp(model_12/dense_12/BiasAdd/ReadVariableOp2X
*model_12/dense_12/BiasAdd_1/ReadVariableOp*model_12/dense_12/BiasAdd_1/ReadVariableOp2R
'model_12/dense_12/MatMul/ReadVariableOp'model_12/dense_12/MatMul/ReadVariableOp2V
)model_12/dense_12/MatMul_1/ReadVariableOp)model_12/dense_12/MatMul_1/ReadVariableOp:Y U
/
_output_shapes
:?????????dd
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:?????????dd
"
_user_specified_name
inputs/1
³

d
E__inference_dropout_12_layer_call_and_return_conditional_losses_23494

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nΫΆ?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????22@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????22@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????22@w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????22@q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????22@a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????22@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????22@:W S
/
_output_shapes
:?????????22@
 
_user_specified_nameinputs

V
:__inference_global_average_pooling2d_6_layer_call_fn_23556

inputs
identityΙ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *^
fYRW
U__inference_global_average_pooling2d_6_layer_call_and_return_conditional_losses_22463i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs

g
K__inference_max_pooling2d_13_layer_call_and_return_conditional_losses_22450

inputs
identity’
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
κ

)__inference_conv2d_13_layer_call_fn_23503

inputs!
unknown:@@
	unknown_0:@
identity’StatefulPartitionedCallα
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????22@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_13_layer_call_and_return_conditional_losses_22509w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????22@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????22@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????22@
 
_user_specified_nameinputs
	

(__inference_model_12_layer_call_fn_22707
input_21!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@
	unknown_3:@0
	unknown_4:0
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_21unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????0*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_model_12_layer_call_and_return_conditional_losses_22675o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????0`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????dd: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:?????????dd
"
_user_specified_name
input_21
Ζ	
τ
C__inference_dense_12_layer_call_and_return_conditional_losses_22534

inputs0
matmul_readvariableop_resource:@0-
biasadd_readvariableop_resource:0
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@0*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????0r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????0_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????0w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
Ι

?
#__inference_signature_wrapper_23269
input_19
input_20!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@
	unknown_3:@0
	unknown_4:0
	unknown_5:
	unknown_6:
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_19input_20unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__wrapped_model_22429o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:?????????dd:?????????dd: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:?????????dd
"
_user_specified_name
input_19:YU
/
_output_shapes
:?????????dd
"
_user_specified_name
input_20
³

d
E__inference_dropout_13_layer_call_and_return_conditional_losses_23551

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nΫΆ?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????@w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????@a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs

ύ
D__inference_conv2d_12_layer_call_and_return_conditional_losses_23457

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity’BiasAdd/ReadVariableOp’Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????dd@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????dd@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????dd@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????dd@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????dd: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????dd
 
_user_specified_nameinputs


m
C__inference_lambda_6_layer_call_and_return_conditional_losses_22798

inputs
inputs_1
identityN
subSubinputsinputs_1*
T0*'
_output_shapes
:?????????0K
SquareSquaresub:z:0*
T0*'
_output_shapes
:?????????0W
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :y
SumSum
Square:y:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(N
	Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΏΦ3f
MaximumMaximumSum:output:0Maximum/y:output:0*
T0*'
_output_shapes
:?????????J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    c
	Maximum_1MaximumMaximum:z:0Const:output:0*
T0*'
_output_shapes
:?????????M
SqrtSqrtMaximum_1:z:0*
T0*'
_output_shapes
:?????????P
IdentityIdentitySqrt:y:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????0:?????????0:O K
'
_output_shapes
:?????????0
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????0
 
_user_specified_nameinputs
Ϊ
ν
C__inference_model_13_layer_call_and_return_conditional_losses_23002
input_19
input_20(
model_12_22975:@
model_12_22977:@(
model_12_22979:@@
model_12_22981:@ 
model_12_22983:@0
model_12_22985:0 
dense_13_22996:
dense_13_22998:
identity’ dense_13/StatefulPartitionedCall’ model_12/StatefulPartitionedCall’"model_12/StatefulPartitionedCall_1·
 model_12/StatefulPartitionedCallStatefulPartitionedCallinput_19model_12_22975model_12_22977model_12_22979model_12_22981model_12_22983model_12_22985*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????0*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_model_12_layer_call_and_return_conditional_losses_22541Ή
"model_12/StatefulPartitionedCall_1StatefulPartitionedCallinput_20model_12_22975model_12_22977model_12_22979model_12_22981model_12_22983model_12_22985*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????0*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_model_12_layer_call_and_return_conditional_losses_22541
lambda_6/PartitionedCallPartitionedCall)model_12/StatefulPartitionedCall:output:0+model_12/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lambda_6_layer_call_and_return_conditional_losses_22798
 dense_13/StatefulPartitionedCallStatefulPartitionedCall!lambda_6/PartitionedCall:output:0dense_13_22996dense_13_22998*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_13_layer_call_and_return_conditional_losses_22811x
IdentityIdentity)dense_13/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????±
NoOpNoOp!^dense_13/StatefulPartitionedCall!^model_12/StatefulPartitionedCall#^model_12/StatefulPartitionedCall_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:?????????dd:?????????dd: : : : : : : : 2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 model_12/StatefulPartitionedCall model_12/StatefulPartitionedCall2H
"model_12/StatefulPartitionedCall_1"model_12/StatefulPartitionedCall_1:Y U
/
_output_shapes
:?????????dd
"
_user_specified_name
input_19:YU
/
_output_shapes
:?????????dd
"
_user_specified_name
input_20
χ
έ	
C__inference_model_13_layer_call_and_return_conditional_losses_23245
inputs_0
inputs_1K
1model_12_conv2d_12_conv2d_readvariableop_resource:@@
2model_12_conv2d_12_biasadd_readvariableop_resource:@K
1model_12_conv2d_13_conv2d_readvariableop_resource:@@@
2model_12_conv2d_13_biasadd_readvariableop_resource:@B
0model_12_dense_12_matmul_readvariableop_resource:@0?
1model_12_dense_12_biasadd_readvariableop_resource:09
'dense_13_matmul_readvariableop_resource:6
(dense_13_biasadd_readvariableop_resource:
identity’dense_13/BiasAdd/ReadVariableOp’dense_13/MatMul/ReadVariableOp’)model_12/conv2d_12/BiasAdd/ReadVariableOp’+model_12/conv2d_12/BiasAdd_1/ReadVariableOp’(model_12/conv2d_12/Conv2D/ReadVariableOp’*model_12/conv2d_12/Conv2D_1/ReadVariableOp’)model_12/conv2d_13/BiasAdd/ReadVariableOp’+model_12/conv2d_13/BiasAdd_1/ReadVariableOp’(model_12/conv2d_13/Conv2D/ReadVariableOp’*model_12/conv2d_13/Conv2D_1/ReadVariableOp’(model_12/dense_12/BiasAdd/ReadVariableOp’*model_12/dense_12/BiasAdd_1/ReadVariableOp’'model_12/dense_12/MatMul/ReadVariableOp’)model_12/dense_12/MatMul_1/ReadVariableOp’
(model_12/conv2d_12/Conv2D/ReadVariableOpReadVariableOp1model_12_conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Α
model_12/conv2d_12/Conv2DConv2Dinputs_00model_12/conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????dd@*
paddingSAME*
strides

)model_12/conv2d_12/BiasAdd/ReadVariableOpReadVariableOp2model_12_conv2d_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ά
model_12/conv2d_12/BiasAddBiasAdd"model_12/conv2d_12/Conv2D:output:01model_12/conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????dd@~
model_12/conv2d_12/ReluRelu#model_12/conv2d_12/BiasAdd:output:0*
T0*/
_output_shapes
:?????????dd@ΐ
!model_12/max_pooling2d_12/MaxPoolMaxPool%model_12/conv2d_12/Relu:activations:0*/
_output_shapes
:?????????22@*
ksize
*
paddingVALID*
strides
f
!model_12/dropout_12/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nΫΆ?Έ
model_12/dropout_12/dropout/MulMul*model_12/max_pooling2d_12/MaxPool:output:0*model_12/dropout_12/dropout/Const:output:0*
T0*/
_output_shapes
:?????????22@{
!model_12/dropout_12/dropout/ShapeShape*model_12/max_pooling2d_12/MaxPool:output:0*
T0*
_output_shapes
:Ό
8model_12/dropout_12/dropout/random_uniform/RandomUniformRandomUniform*model_12/dropout_12/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????22@*
dtype0o
*model_12/dropout_12/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>κ
(model_12/dropout_12/dropout/GreaterEqualGreaterEqualAmodel_12/dropout_12/dropout/random_uniform/RandomUniform:output:03model_12/dropout_12/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????22@
 model_12/dropout_12/dropout/CastCast,model_12/dropout_12/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????22@­
!model_12/dropout_12/dropout/Mul_1Mul#model_12/dropout_12/dropout/Mul:z:0$model_12/dropout_12/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????22@’
(model_12/conv2d_13/Conv2D/ReadVariableOpReadVariableOp1model_12_conv2d_13_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0ή
model_12/conv2d_13/Conv2DConv2D%model_12/dropout_12/dropout/Mul_1:z:00model_12/conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????22@*
paddingSAME*
strides

)model_12/conv2d_13/BiasAdd/ReadVariableOpReadVariableOp2model_12_conv2d_13_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ά
model_12/conv2d_13/BiasAddBiasAdd"model_12/conv2d_13/Conv2D:output:01model_12/conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????22@~
model_12/conv2d_13/ReluRelu#model_12/conv2d_13/BiasAdd:output:0*
T0*/
_output_shapes
:?????????22@ΐ
!model_12/max_pooling2d_13/MaxPoolMaxPool%model_12/conv2d_13/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
f
!model_12/dropout_13/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nΫΆ?Έ
model_12/dropout_13/dropout/MulMul*model_12/max_pooling2d_13/MaxPool:output:0*model_12/dropout_13/dropout/Const:output:0*
T0*/
_output_shapes
:?????????@{
!model_12/dropout_13/dropout/ShapeShape*model_12/max_pooling2d_13/MaxPool:output:0*
T0*
_output_shapes
:Ό
8model_12/dropout_13/dropout/random_uniform/RandomUniformRandomUniform*model_12/dropout_13/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????@*
dtype0o
*model_12/dropout_13/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>κ
(model_12/dropout_13/dropout/GreaterEqualGreaterEqualAmodel_12/dropout_13/dropout/random_uniform/RandomUniform:output:03model_12/dropout_13/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????@
 model_12/dropout_13/dropout/CastCast,model_12/dropout_13/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@­
!model_12/dropout_13/dropout/Mul_1Mul#model_12/dropout_13/dropout/Mul:z:0$model_12/dropout_13/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????@
:model_12/global_average_pooling2d_6/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      Ξ
(model_12/global_average_pooling2d_6/MeanMean%model_12/dropout_13/dropout/Mul_1:z:0Cmodel_12/global_average_pooling2d_6/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????@
'model_12/dense_12/MatMul/ReadVariableOpReadVariableOp0model_12_dense_12_matmul_readvariableop_resource*
_output_shapes

:@0*
dtype0Έ
model_12/dense_12/MatMulMatMul1model_12/global_average_pooling2d_6/Mean:output:0/model_12/dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????0
(model_12/dense_12/BiasAdd/ReadVariableOpReadVariableOp1model_12_dense_12_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0¬
model_12/dense_12/BiasAddBiasAdd"model_12/dense_12/MatMul:product:00model_12/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????0€
*model_12/conv2d_12/Conv2D_1/ReadVariableOpReadVariableOp1model_12_conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Ε
model_12/conv2d_12/Conv2D_1Conv2Dinputs_12model_12/conv2d_12/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????dd@*
paddingSAME*
strides

+model_12/conv2d_12/BiasAdd_1/ReadVariableOpReadVariableOp2model_12_conv2d_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ό
model_12/conv2d_12/BiasAdd_1BiasAdd$model_12/conv2d_12/Conv2D_1:output:03model_12/conv2d_12/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????dd@
model_12/conv2d_12/Relu_1Relu%model_12/conv2d_12/BiasAdd_1:output:0*
T0*/
_output_shapes
:?????????dd@Δ
#model_12/max_pooling2d_12/MaxPool_1MaxPool'model_12/conv2d_12/Relu_1:activations:0*/
_output_shapes
:?????????22@*
ksize
*
paddingVALID*
strides
h
#model_12/dropout_12/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nΫΆ?Ύ
!model_12/dropout_12/dropout_1/MulMul,model_12/max_pooling2d_12/MaxPool_1:output:0,model_12/dropout_12/dropout_1/Const:output:0*
T0*/
_output_shapes
:?????????22@
#model_12/dropout_12/dropout_1/ShapeShape,model_12/max_pooling2d_12/MaxPool_1:output:0*
T0*
_output_shapes
:ΐ
:model_12/dropout_12/dropout_1/random_uniform/RandomUniformRandomUniform,model_12/dropout_12/dropout_1/Shape:output:0*
T0*/
_output_shapes
:?????????22@*
dtype0q
,model_12/dropout_12/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>π
*model_12/dropout_12/dropout_1/GreaterEqualGreaterEqualCmodel_12/dropout_12/dropout_1/random_uniform/RandomUniform:output:05model_12/dropout_12/dropout_1/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????22@£
"model_12/dropout_12/dropout_1/CastCast.model_12/dropout_12/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????22@³
#model_12/dropout_12/dropout_1/Mul_1Mul%model_12/dropout_12/dropout_1/Mul:z:0&model_12/dropout_12/dropout_1/Cast:y:0*
T0*/
_output_shapes
:?????????22@€
*model_12/conv2d_13/Conv2D_1/ReadVariableOpReadVariableOp1model_12_conv2d_13_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0δ
model_12/conv2d_13/Conv2D_1Conv2D'model_12/dropout_12/dropout_1/Mul_1:z:02model_12/conv2d_13/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????22@*
paddingSAME*
strides

+model_12/conv2d_13/BiasAdd_1/ReadVariableOpReadVariableOp2model_12_conv2d_13_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ό
model_12/conv2d_13/BiasAdd_1BiasAdd$model_12/conv2d_13/Conv2D_1:output:03model_12/conv2d_13/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????22@
model_12/conv2d_13/Relu_1Relu%model_12/conv2d_13/BiasAdd_1:output:0*
T0*/
_output_shapes
:?????????22@Δ
#model_12/max_pooling2d_13/MaxPool_1MaxPool'model_12/conv2d_13/Relu_1:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
h
#model_12/dropout_13/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nΫΆ?Ύ
!model_12/dropout_13/dropout_1/MulMul,model_12/max_pooling2d_13/MaxPool_1:output:0,model_12/dropout_13/dropout_1/Const:output:0*
T0*/
_output_shapes
:?????????@
#model_12/dropout_13/dropout_1/ShapeShape,model_12/max_pooling2d_13/MaxPool_1:output:0*
T0*
_output_shapes
:ΐ
:model_12/dropout_13/dropout_1/random_uniform/RandomUniformRandomUniform,model_12/dropout_13/dropout_1/Shape:output:0*
T0*/
_output_shapes
:?????????@*
dtype0q
,model_12/dropout_13/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>π
*model_12/dropout_13/dropout_1/GreaterEqualGreaterEqualCmodel_12/dropout_13/dropout_1/random_uniform/RandomUniform:output:05model_12/dropout_13/dropout_1/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????@£
"model_12/dropout_13/dropout_1/CastCast.model_12/dropout_13/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@³
#model_12/dropout_13/dropout_1/Mul_1Mul%model_12/dropout_13/dropout_1/Mul:z:0&model_12/dropout_13/dropout_1/Cast:y:0*
T0*/
_output_shapes
:?????????@
<model_12/global_average_pooling2d_6/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      Τ
*model_12/global_average_pooling2d_6/Mean_1Mean'model_12/dropout_13/dropout_1/Mul_1:z:0Emodel_12/global_average_pooling2d_6/Mean_1/reduction_indices:output:0*
T0*'
_output_shapes
:?????????@
)model_12/dense_12/MatMul_1/ReadVariableOpReadVariableOp0model_12_dense_12_matmul_readvariableop_resource*
_output_shapes

:@0*
dtype0Ύ
model_12/dense_12/MatMul_1MatMul3model_12/global_average_pooling2d_6/Mean_1:output:01model_12/dense_12/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????0
*model_12/dense_12/BiasAdd_1/ReadVariableOpReadVariableOp1model_12_dense_12_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0²
model_12/dense_12/BiasAdd_1BiasAdd$model_12/dense_12/MatMul_1:product:02model_12/dense_12/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????0
lambda_6/subSub"model_12/dense_12/BiasAdd:output:0$model_12/dense_12/BiasAdd_1:output:0*
T0*'
_output_shapes
:?????????0]
lambda_6/SquareSquarelambda_6/sub:z:0*
T0*'
_output_shapes
:?????????0`
lambda_6/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :
lambda_6/SumSumlambda_6/Square:y:0'lambda_6/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(W
lambda_6/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΏΦ3
lambda_6/MaximumMaximumlambda_6/Sum:output:0lambda_6/Maximum/y:output:0*
T0*'
_output_shapes
:?????????S
lambda_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ~
lambda_6/Maximum_1Maximumlambda_6/Maximum:z:0lambda_6/Const:output:0*
T0*'
_output_shapes
:?????????_
lambda_6/SqrtSqrtlambda_6/Maximum_1:z:0*
T0*'
_output_shapes
:?????????
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_13/MatMulMatMullambda_6/Sqrt:y:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????h
dense_13/SigmoidSigmoiddense_13/BiasAdd:output:0*
T0*'
_output_shapes
:?????????c
IdentityIdentitydense_13/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????
NoOpNoOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp*^model_12/conv2d_12/BiasAdd/ReadVariableOp,^model_12/conv2d_12/BiasAdd_1/ReadVariableOp)^model_12/conv2d_12/Conv2D/ReadVariableOp+^model_12/conv2d_12/Conv2D_1/ReadVariableOp*^model_12/conv2d_13/BiasAdd/ReadVariableOp,^model_12/conv2d_13/BiasAdd_1/ReadVariableOp)^model_12/conv2d_13/Conv2D/ReadVariableOp+^model_12/conv2d_13/Conv2D_1/ReadVariableOp)^model_12/dense_12/BiasAdd/ReadVariableOp+^model_12/dense_12/BiasAdd_1/ReadVariableOp(^model_12/dense_12/MatMul/ReadVariableOp*^model_12/dense_12/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:?????????dd:?????????dd: : : : : : : : 2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2V
)model_12/conv2d_12/BiasAdd/ReadVariableOp)model_12/conv2d_12/BiasAdd/ReadVariableOp2Z
+model_12/conv2d_12/BiasAdd_1/ReadVariableOp+model_12/conv2d_12/BiasAdd_1/ReadVariableOp2T
(model_12/conv2d_12/Conv2D/ReadVariableOp(model_12/conv2d_12/Conv2D/ReadVariableOp2X
*model_12/conv2d_12/Conv2D_1/ReadVariableOp*model_12/conv2d_12/Conv2D_1/ReadVariableOp2V
)model_12/conv2d_13/BiasAdd/ReadVariableOp)model_12/conv2d_13/BiasAdd/ReadVariableOp2Z
+model_12/conv2d_13/BiasAdd_1/ReadVariableOp+model_12/conv2d_13/BiasAdd_1/ReadVariableOp2T
(model_12/conv2d_13/Conv2D/ReadVariableOp(model_12/conv2d_13/Conv2D/ReadVariableOp2X
*model_12/conv2d_13/Conv2D_1/ReadVariableOp*model_12/conv2d_13/Conv2D_1/ReadVariableOp2T
(model_12/dense_12/BiasAdd/ReadVariableOp(model_12/dense_12/BiasAdd/ReadVariableOp2X
*model_12/dense_12/BiasAdd_1/ReadVariableOp*model_12/dense_12/BiasAdd_1/ReadVariableOp2R
'model_12/dense_12/MatMul/ReadVariableOp'model_12/dense_12/MatMul/ReadVariableOp2V
)model_12/dense_12/MatMul_1/ReadVariableOp)model_12/dense_12/MatMul_1/ReadVariableOp:Y U
/
_output_shapes
:?????????dd
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:?????????dd
"
_user_specified_name
inputs/1
Έ
L
0__inference_max_pooling2d_13_layer_call_fn_23519

inputs
identityΩ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_max_pooling2d_13_layer_call_and_return_conditional_losses_22450
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
	

(__inference_model_12_layer_call_fn_23303

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@
	unknown_3:@0
	unknown_4:0
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????0*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_model_12_layer_call_and_return_conditional_losses_22675o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????0`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????dd: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????dd
 
_user_specified_nameinputs
κ

)__inference_conv2d_12_layer_call_fn_23446

inputs!
unknown:@
	unknown_0:@
identity’StatefulPartitionedCallα
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????dd@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_12_layer_call_and_return_conditional_losses_22484w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????dd@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????dd: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????dd
 
_user_specified_nameinputs

g
K__inference_max_pooling2d_13_layer_call_and_return_conditional_losses_23524

inputs
identity’
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
	

(__inference_model_12_layer_call_fn_23286

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@
	unknown_3:@0
	unknown_4:0
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????0*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_model_12_layer_call_and_return_conditional_losses_22541o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????0`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????dd: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????dd
 
_user_specified_nameinputs
Ζ	
τ
C__inference_dense_12_layer_call_and_return_conditional_losses_23581

inputs0
matmul_readvariableop_resource:@0-
biasadd_readvariableop_resource:0
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@0*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????0r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????0_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????0w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
ρ 
‘
C__inference_model_12_layer_call_and_return_conditional_losses_22541

inputs)
conv2d_12_22485:@
conv2d_12_22487:@)
conv2d_13_22510:@@
conv2d_13_22512:@ 
dense_12_22535:@0
dense_12_22537:0
identity’!conv2d_12/StatefulPartitionedCall’!conv2d_13/StatefulPartitionedCall’ dense_12/StatefulPartitionedCallω
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_12_22485conv2d_12_22487*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????dd@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_12_layer_call_and_return_conditional_losses_22484σ
 max_pooling2d_12/PartitionedCallPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????22@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_22438ζ
dropout_12/PartitionedCallPartitionedCall)max_pooling2d_12/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????22@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_12_layer_call_and_return_conditional_losses_22496
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall#dropout_12/PartitionedCall:output:0conv2d_13_22510conv2d_13_22512*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????22@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_13_layer_call_and_return_conditional_losses_22509σ
 max_pooling2d_13/PartitionedCallPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_max_pooling2d_13_layer_call_and_return_conditional_losses_22450ζ
dropout_13/PartitionedCallPartitionedCall)max_pooling2d_13/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_13_layer_call_and_return_conditional_losses_22521ψ
*global_average_pooling2d_6/PartitionedCallPartitionedCall#dropout_13/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *^
fYRW
U__inference_global_average_pooling2d_6_layer_call_and_return_conditional_losses_22463
 dense_12/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling2d_6/PartitionedCall:output:0dense_12_22535dense_12_22537*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_12_layer_call_and_return_conditional_losses_22534x
IdentityIdentity)dense_12/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????0±
NoOpNoOp"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????dd: : : : : : 2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall:W S
/
_output_shapes
:?????????dd
 
_user_specified_nameinputs

g
K__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_23467

inputs
identity’
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs

c
*__inference_dropout_13_layer_call_fn_23534

inputs
identity’StatefulPartitionedCallΘ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_13_layer_call_and_return_conditional_losses_22586w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
Ά
q
U__inference_global_average_pooling2d_6_layer_call_and_return_conditional_losses_22463

inputs
identityg
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
ρ

Χ
(__inference_model_13_layer_call_fn_22971
input_19
input_20!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@
	unknown_3:@0
	unknown_4:0
	unknown_5:
	unknown_6:
identity’StatefulPartitionedCall³
StatefulPartitionedCallStatefulPartitionedCallinput_19input_20unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_model_13_layer_call_and_return_conditional_losses_22930o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:?????????dd:?????????dd: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:?????????dd
"
_user_specified_name
input_19:YU
/
_output_shapes
:?????????dd
"
_user_specified_name
input_20
­k

 __inference__wrapped_model_22429
input_19
input_20T
:model_13_model_12_conv2d_12_conv2d_readvariableop_resource:@I
;model_13_model_12_conv2d_12_biasadd_readvariableop_resource:@T
:model_13_model_12_conv2d_13_conv2d_readvariableop_resource:@@I
;model_13_model_12_conv2d_13_biasadd_readvariableop_resource:@K
9model_13_model_12_dense_12_matmul_readvariableop_resource:@0H
:model_13_model_12_dense_12_biasadd_readvariableop_resource:0B
0model_13_dense_13_matmul_readvariableop_resource:?
1model_13_dense_13_biasadd_readvariableop_resource:
identity’(model_13/dense_13/BiasAdd/ReadVariableOp’'model_13/dense_13/MatMul/ReadVariableOp’2model_13/model_12/conv2d_12/BiasAdd/ReadVariableOp’4model_13/model_12/conv2d_12/BiasAdd_1/ReadVariableOp’1model_13/model_12/conv2d_12/Conv2D/ReadVariableOp’3model_13/model_12/conv2d_12/Conv2D_1/ReadVariableOp’2model_13/model_12/conv2d_13/BiasAdd/ReadVariableOp’4model_13/model_12/conv2d_13/BiasAdd_1/ReadVariableOp’1model_13/model_12/conv2d_13/Conv2D/ReadVariableOp’3model_13/model_12/conv2d_13/Conv2D_1/ReadVariableOp’1model_13/model_12/dense_12/BiasAdd/ReadVariableOp’3model_13/model_12/dense_12/BiasAdd_1/ReadVariableOp’0model_13/model_12/dense_12/MatMul/ReadVariableOp’2model_13/model_12/dense_12/MatMul_1/ReadVariableOp΄
1model_13/model_12/conv2d_12/Conv2D/ReadVariableOpReadVariableOp:model_13_model_12_conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Σ
"model_13/model_12/conv2d_12/Conv2DConv2Dinput_199model_13/model_12/conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????dd@*
paddingSAME*
strides
ͺ
2model_13/model_12/conv2d_12/BiasAdd/ReadVariableOpReadVariableOp;model_13_model_12_conv2d_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ρ
#model_13/model_12/conv2d_12/BiasAddBiasAdd+model_13/model_12/conv2d_12/Conv2D:output:0:model_13/model_12/conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????dd@
 model_13/model_12/conv2d_12/ReluRelu,model_13/model_12/conv2d_12/BiasAdd:output:0*
T0*/
_output_shapes
:?????????dd@?
*model_13/model_12/max_pooling2d_12/MaxPoolMaxPool.model_13/model_12/conv2d_12/Relu:activations:0*/
_output_shapes
:?????????22@*
ksize
*
paddingVALID*
strides
 
%model_13/model_12/dropout_12/IdentityIdentity3model_13/model_12/max_pooling2d_12/MaxPool:output:0*
T0*/
_output_shapes
:?????????22@΄
1model_13/model_12/conv2d_13/Conv2D/ReadVariableOpReadVariableOp:model_13_model_12_conv2d_13_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0ω
"model_13/model_12/conv2d_13/Conv2DConv2D.model_13/model_12/dropout_12/Identity:output:09model_13/model_12/conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????22@*
paddingSAME*
strides
ͺ
2model_13/model_12/conv2d_13/BiasAdd/ReadVariableOpReadVariableOp;model_13_model_12_conv2d_13_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ρ
#model_13/model_12/conv2d_13/BiasAddBiasAdd+model_13/model_12/conv2d_13/Conv2D:output:0:model_13/model_12/conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????22@
 model_13/model_12/conv2d_13/ReluRelu,model_13/model_12/conv2d_13/BiasAdd:output:0*
T0*/
_output_shapes
:?????????22@?
*model_13/model_12/max_pooling2d_13/MaxPoolMaxPool.model_13/model_12/conv2d_13/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
 
%model_13/model_12/dropout_13/IdentityIdentity3model_13/model_12/max_pooling2d_13/MaxPool:output:0*
T0*/
_output_shapes
:?????????@
Cmodel_13/model_12/global_average_pooling2d_6/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ι
1model_13/model_12/global_average_pooling2d_6/MeanMean.model_13/model_12/dropout_13/Identity:output:0Lmodel_13/model_12/global_average_pooling2d_6/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????@ͺ
0model_13/model_12/dense_12/MatMul/ReadVariableOpReadVariableOp9model_13_model_12_dense_12_matmul_readvariableop_resource*
_output_shapes

:@0*
dtype0Σ
!model_13/model_12/dense_12/MatMulMatMul:model_13/model_12/global_average_pooling2d_6/Mean:output:08model_13/model_12/dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????0¨
1model_13/model_12/dense_12/BiasAdd/ReadVariableOpReadVariableOp:model_13_model_12_dense_12_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0Η
"model_13/model_12/dense_12/BiasAddBiasAdd+model_13/model_12/dense_12/MatMul:product:09model_13/model_12/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????0Ά
3model_13/model_12/conv2d_12/Conv2D_1/ReadVariableOpReadVariableOp:model_13_model_12_conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Χ
$model_13/model_12/conv2d_12/Conv2D_1Conv2Dinput_20;model_13/model_12/conv2d_12/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????dd@*
paddingSAME*
strides
¬
4model_13/model_12/conv2d_12/BiasAdd_1/ReadVariableOpReadVariableOp;model_13_model_12_conv2d_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Χ
%model_13/model_12/conv2d_12/BiasAdd_1BiasAdd-model_13/model_12/conv2d_12/Conv2D_1:output:0<model_13/model_12/conv2d_12/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????dd@
"model_13/model_12/conv2d_12/Relu_1Relu.model_13/model_12/conv2d_12/BiasAdd_1:output:0*
T0*/
_output_shapes
:?????????dd@Φ
,model_13/model_12/max_pooling2d_12/MaxPool_1MaxPool0model_13/model_12/conv2d_12/Relu_1:activations:0*/
_output_shapes
:?????????22@*
ksize
*
paddingVALID*
strides
€
'model_13/model_12/dropout_12/Identity_1Identity5model_13/model_12/max_pooling2d_12/MaxPool_1:output:0*
T0*/
_output_shapes
:?????????22@Ά
3model_13/model_12/conv2d_13/Conv2D_1/ReadVariableOpReadVariableOp:model_13_model_12_conv2d_13_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
$model_13/model_12/conv2d_13/Conv2D_1Conv2D0model_13/model_12/dropout_12/Identity_1:output:0;model_13/model_12/conv2d_13/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????22@*
paddingSAME*
strides
¬
4model_13/model_12/conv2d_13/BiasAdd_1/ReadVariableOpReadVariableOp;model_13_model_12_conv2d_13_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Χ
%model_13/model_12/conv2d_13/BiasAdd_1BiasAdd-model_13/model_12/conv2d_13/Conv2D_1:output:0<model_13/model_12/conv2d_13/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????22@
"model_13/model_12/conv2d_13/Relu_1Relu.model_13/model_12/conv2d_13/BiasAdd_1:output:0*
T0*/
_output_shapes
:?????????22@Φ
,model_13/model_12/max_pooling2d_13/MaxPool_1MaxPool0model_13/model_12/conv2d_13/Relu_1:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
€
'model_13/model_12/dropout_13/Identity_1Identity5model_13/model_12/max_pooling2d_13/MaxPool_1:output:0*
T0*/
_output_shapes
:?????????@
Emodel_13/model_12/global_average_pooling2d_6/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ο
3model_13/model_12/global_average_pooling2d_6/Mean_1Mean0model_13/model_12/dropout_13/Identity_1:output:0Nmodel_13/model_12/global_average_pooling2d_6/Mean_1/reduction_indices:output:0*
T0*'
_output_shapes
:?????????@¬
2model_13/model_12/dense_12/MatMul_1/ReadVariableOpReadVariableOp9model_13_model_12_dense_12_matmul_readvariableop_resource*
_output_shapes

:@0*
dtype0Ω
#model_13/model_12/dense_12/MatMul_1MatMul<model_13/model_12/global_average_pooling2d_6/Mean_1:output:0:model_13/model_12/dense_12/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????0ͺ
3model_13/model_12/dense_12/BiasAdd_1/ReadVariableOpReadVariableOp:model_13_model_12_dense_12_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0Ν
$model_13/model_12/dense_12/BiasAdd_1BiasAdd-model_13/model_12/dense_12/MatMul_1:product:0;model_13/model_12/dense_12/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????0ͺ
model_13/lambda_6/subSub+model_13/model_12/dense_12/BiasAdd:output:0-model_13/model_12/dense_12/BiasAdd_1:output:0*
T0*'
_output_shapes
:?????????0o
model_13/lambda_6/SquareSquaremodel_13/lambda_6/sub:z:0*
T0*'
_output_shapes
:?????????0i
'model_13/lambda_6/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :―
model_13/lambda_6/SumSummodel_13/lambda_6/Square:y:00model_13/lambda_6/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(`
model_13/lambda_6/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΏΦ3
model_13/lambda_6/MaximumMaximummodel_13/lambda_6/Sum:output:0$model_13/lambda_6/Maximum/y:output:0*
T0*'
_output_shapes
:?????????\
model_13/lambda_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
model_13/lambda_6/Maximum_1Maximummodel_13/lambda_6/Maximum:z:0 model_13/lambda_6/Const:output:0*
T0*'
_output_shapes
:?????????q
model_13/lambda_6/SqrtSqrtmodel_13/lambda_6/Maximum_1:z:0*
T0*'
_output_shapes
:?????????
'model_13/dense_13/MatMul/ReadVariableOpReadVariableOp0model_13_dense_13_matmul_readvariableop_resource*
_output_shapes

:*
dtype0‘
model_13/dense_13/MatMulMatMulmodel_13/lambda_6/Sqrt:y:0/model_13/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
(model_13/dense_13/BiasAdd/ReadVariableOpReadVariableOp1model_13_dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¬
model_13/dense_13/BiasAddBiasAdd"model_13/dense_13/MatMul:product:00model_13/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????z
model_13/dense_13/SigmoidSigmoid"model_13/dense_13/BiasAdd:output:0*
T0*'
_output_shapes
:?????????l
IdentityIdentitymodel_13/dense_13/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????
NoOpNoOp)^model_13/dense_13/BiasAdd/ReadVariableOp(^model_13/dense_13/MatMul/ReadVariableOp3^model_13/model_12/conv2d_12/BiasAdd/ReadVariableOp5^model_13/model_12/conv2d_12/BiasAdd_1/ReadVariableOp2^model_13/model_12/conv2d_12/Conv2D/ReadVariableOp4^model_13/model_12/conv2d_12/Conv2D_1/ReadVariableOp3^model_13/model_12/conv2d_13/BiasAdd/ReadVariableOp5^model_13/model_12/conv2d_13/BiasAdd_1/ReadVariableOp2^model_13/model_12/conv2d_13/Conv2D/ReadVariableOp4^model_13/model_12/conv2d_13/Conv2D_1/ReadVariableOp2^model_13/model_12/dense_12/BiasAdd/ReadVariableOp4^model_13/model_12/dense_12/BiasAdd_1/ReadVariableOp1^model_13/model_12/dense_12/MatMul/ReadVariableOp3^model_13/model_12/dense_12/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:?????????dd:?????????dd: : : : : : : : 2T
(model_13/dense_13/BiasAdd/ReadVariableOp(model_13/dense_13/BiasAdd/ReadVariableOp2R
'model_13/dense_13/MatMul/ReadVariableOp'model_13/dense_13/MatMul/ReadVariableOp2h
2model_13/model_12/conv2d_12/BiasAdd/ReadVariableOp2model_13/model_12/conv2d_12/BiasAdd/ReadVariableOp2l
4model_13/model_12/conv2d_12/BiasAdd_1/ReadVariableOp4model_13/model_12/conv2d_12/BiasAdd_1/ReadVariableOp2f
1model_13/model_12/conv2d_12/Conv2D/ReadVariableOp1model_13/model_12/conv2d_12/Conv2D/ReadVariableOp2j
3model_13/model_12/conv2d_12/Conv2D_1/ReadVariableOp3model_13/model_12/conv2d_12/Conv2D_1/ReadVariableOp2h
2model_13/model_12/conv2d_13/BiasAdd/ReadVariableOp2model_13/model_12/conv2d_13/BiasAdd/ReadVariableOp2l
4model_13/model_12/conv2d_13/BiasAdd_1/ReadVariableOp4model_13/model_12/conv2d_13/BiasAdd_1/ReadVariableOp2f
1model_13/model_12/conv2d_13/Conv2D/ReadVariableOp1model_13/model_12/conv2d_13/Conv2D/ReadVariableOp2j
3model_13/model_12/conv2d_13/Conv2D_1/ReadVariableOp3model_13/model_12/conv2d_13/Conv2D_1/ReadVariableOp2f
1model_13/model_12/dense_12/BiasAdd/ReadVariableOp1model_13/model_12/dense_12/BiasAdd/ReadVariableOp2j
3model_13/model_12/dense_12/BiasAdd_1/ReadVariableOp3model_13/model_12/dense_12/BiasAdd_1/ReadVariableOp2d
0model_13/model_12/dense_12/MatMul/ReadVariableOp0model_13/model_12/dense_12/MatMul/ReadVariableOp2h
2model_13/model_12/dense_12/MatMul_1/ReadVariableOp2model_13/model_12/dense_12/MatMul_1/ReadVariableOp:Y U
/
_output_shapes
:?????????dd
"
_user_specified_name
input_19:YU
/
_output_shapes
:?????????dd
"
_user_specified_name
input_20
ψ
c
E__inference_dropout_13_layer_call_and_return_conditional_losses_22521

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????@c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
Ϊ
ν
C__inference_model_13_layer_call_and_return_conditional_losses_23033
input_19
input_20(
model_12_23006:@
model_12_23008:@(
model_12_23010:@@
model_12_23012:@ 
model_12_23014:@0
model_12_23016:0 
dense_13_23027:
dense_13_23029:
identity’ dense_13/StatefulPartitionedCall’ model_12/StatefulPartitionedCall’"model_12/StatefulPartitionedCall_1·
 model_12/StatefulPartitionedCallStatefulPartitionedCallinput_19model_12_23006model_12_23008model_12_23010model_12_23012model_12_23014model_12_23016*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????0*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_model_12_layer_call_and_return_conditional_losses_22675Ή
"model_12/StatefulPartitionedCall_1StatefulPartitionedCallinput_20model_12_23006model_12_23008model_12_23010model_12_23012model_12_23014model_12_23016*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????0*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_model_12_layer_call_and_return_conditional_losses_22675
lambda_6/PartitionedCallPartitionedCall)model_12/StatefulPartitionedCall:output:0+model_12/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lambda_6_layer_call_and_return_conditional_losses_22871
 dense_13/StatefulPartitionedCallStatefulPartitionedCall!lambda_6/PartitionedCall:output:0dense_13_23027dense_13_23029*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_13_layer_call_and_return_conditional_losses_22811x
IdentityIdentity)dense_13/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????±
NoOpNoOp!^dense_13/StatefulPartitionedCall!^model_12/StatefulPartitionedCall#^model_12/StatefulPartitionedCall_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:?????????dd:?????????dd: : : : : : : : 2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 model_12/StatefulPartitionedCall model_12/StatefulPartitionedCall2H
"model_12/StatefulPartitionedCall_1"model_12/StatefulPartitionedCall_1:Y U
/
_output_shapes
:?????????dd
"
_user_specified_name
input_19:YU
/
_output_shapes
:?????????dd
"
_user_specified_name
input_20
	

(__inference_model_12_layer_call_fn_22556
input_21!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@
	unknown_3:@0
	unknown_4:0
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_21unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????0*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_model_12_layer_call_and_return_conditional_losses_22541o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????0`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????dd: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:?????????dd
"
_user_specified_name
input_21
Β

!__inference__traced_restore_23813
file_prefix2
 assignvariableop_dense_13_kernel:.
 assignvariableop_1_dense_13_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: =
#assignvariableop_7_conv2d_12_kernel:@/
!assignvariableop_8_conv2d_12_bias:@=
#assignvariableop_9_conv2d_13_kernel:@@0
"assignvariableop_10_conv2d_13_bias:@5
#assignvariableop_11_dense_12_kernel:@0/
!assignvariableop_12_dense_12_bias:0#
assignvariableop_13_total: #
assignvariableop_14_count: %
assignvariableop_15_total_1: %
assignvariableop_16_count_1: <
*assignvariableop_17_adam_dense_13_kernel_m:6
(assignvariableop_18_adam_dense_13_bias_m:E
+assignvariableop_19_adam_conv2d_12_kernel_m:@7
)assignvariableop_20_adam_conv2d_12_bias_m:@E
+assignvariableop_21_adam_conv2d_13_kernel_m:@@7
)assignvariableop_22_adam_conv2d_13_bias_m:@<
*assignvariableop_23_adam_dense_12_kernel_m:@06
(assignvariableop_24_adam_dense_12_bias_m:0<
*assignvariableop_25_adam_dense_13_kernel_v:6
(assignvariableop_26_adam_dense_13_bias_v:E
+assignvariableop_27_adam_conv2d_12_kernel_v:@7
)assignvariableop_28_adam_conv2d_12_bias_v:@E
+assignvariableop_29_adam_conv2d_13_kernel_v:@@7
)assignvariableop_30_adam_conv2d_13_bias_v:@<
*assignvariableop_31_adam_dense_12_kernel_v:@06
(assignvariableop_32_adam_dense_12_bias_v:0
identity_34’AssignVariableOp’AssignVariableOp_1’AssignVariableOp_10’AssignVariableOp_11’AssignVariableOp_12’AssignVariableOp_13’AssignVariableOp_14’AssignVariableOp_15’AssignVariableOp_16’AssignVariableOp_17’AssignVariableOp_18’AssignVariableOp_19’AssignVariableOp_2’AssignVariableOp_20’AssignVariableOp_21’AssignVariableOp_22’AssignVariableOp_23’AssignVariableOp_24’AssignVariableOp_25’AssignVariableOp_26’AssignVariableOp_27’AssignVariableOp_28’AssignVariableOp_29’AssignVariableOp_3’AssignVariableOp_30’AssignVariableOp_31’AssignVariableOp_32’AssignVariableOp_4’AssignVariableOp_5’AssignVariableOp_6’AssignVariableOp_7’AssignVariableOp_8’AssignVariableOp_9€
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*Κ
valueΐB½"B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH΄
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Λ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes
::::::::::::::::::::::::::::::::::*0
dtypes&
$2"	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp assignvariableop_dense_13_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_13_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp#assignvariableop_7_conv2d_12_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp!assignvariableop_8_conv2d_12_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp#assignvariableop_9_conv2d_13_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp"assignvariableop_10_conv2d_13_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOp#assignvariableop_11_dense_12_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp!assignvariableop_12_dense_12_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOpassignvariableop_15_total_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOpassignvariableop_16_count_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp*assignvariableop_17_adam_dense_13_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp(assignvariableop_18_adam_dense_13_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp+assignvariableop_19_adam_conv2d_12_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp)assignvariableop_20_adam_conv2d_12_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp+assignvariableop_21_adam_conv2d_13_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp)assignvariableop_22_adam_conv2d_13_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_dense_12_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_dense_12_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_dense_13_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_dense_13_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp+assignvariableop_27_adam_conv2d_12_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_conv2d_12_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp+assignvariableop_29_adam_conv2d_13_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_conv2d_13_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_dense_12_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_dense_12_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ₯
Identity_33Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_34IdentityIdentity_33:output:0^NoOp_1*
T0*
_output_shapes
: 
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_34Identity_34:output:0*W
_input_shapesF
D: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ͺ4

C__inference_model_12_layer_call_and_return_conditional_losses_23377

inputsB
(conv2d_12_conv2d_readvariableop_resource:@7
)conv2d_12_biasadd_readvariableop_resource:@B
(conv2d_13_conv2d_readvariableop_resource:@@7
)conv2d_13_biasadd_readvariableop_resource:@9
'dense_12_matmul_readvariableop_resource:@06
(dense_12_biasadd_readvariableop_resource:0
identity’ conv2d_12/BiasAdd/ReadVariableOp’conv2d_12/Conv2D/ReadVariableOp’ conv2d_13/BiasAdd/ReadVariableOp’conv2d_13/Conv2D/ReadVariableOp’dense_12/BiasAdd/ReadVariableOp’dense_12/MatMul/ReadVariableOp
conv2d_12/Conv2D/ReadVariableOpReadVariableOp(conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0­
conv2d_12/Conv2DConv2Dinputs'conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????dd@*
paddingSAME*
strides

 conv2d_12/BiasAdd/ReadVariableOpReadVariableOp)conv2d_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_12/BiasAddBiasAddconv2d_12/Conv2D:output:0(conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????dd@l
conv2d_12/ReluReluconv2d_12/BiasAdd:output:0*
T0*/
_output_shapes
:?????????dd@?
max_pooling2d_12/MaxPoolMaxPoolconv2d_12/Relu:activations:0*/
_output_shapes
:?????????22@*
ksize
*
paddingVALID*
strides
]
dropout_12/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nΫΆ?
dropout_12/dropout/MulMul!max_pooling2d_12/MaxPool:output:0!dropout_12/dropout/Const:output:0*
T0*/
_output_shapes
:?????????22@i
dropout_12/dropout/ShapeShape!max_pooling2d_12/MaxPool:output:0*
T0*
_output_shapes
:ͺ
/dropout_12/dropout/random_uniform/RandomUniformRandomUniform!dropout_12/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????22@*
dtype0f
!dropout_12/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>Ο
dropout_12/dropout/GreaterEqualGreaterEqual8dropout_12/dropout/random_uniform/RandomUniform:output:0*dropout_12/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????22@
dropout_12/dropout/CastCast#dropout_12/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????22@
dropout_12/dropout/Mul_1Muldropout_12/dropout/Mul:z:0dropout_12/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????22@
conv2d_13/Conv2D/ReadVariableOpReadVariableOp(conv2d_13_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Γ
conv2d_13/Conv2DConv2Ddropout_12/dropout/Mul_1:z:0'conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????22@*
paddingSAME*
strides

 conv2d_13/BiasAdd/ReadVariableOpReadVariableOp)conv2d_13_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_13/BiasAddBiasAddconv2d_13/Conv2D:output:0(conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????22@l
conv2d_13/ReluReluconv2d_13/BiasAdd:output:0*
T0*/
_output_shapes
:?????????22@?
max_pooling2d_13/MaxPoolMaxPoolconv2d_13/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
]
dropout_13/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nΫΆ?
dropout_13/dropout/MulMul!max_pooling2d_13/MaxPool:output:0!dropout_13/dropout/Const:output:0*
T0*/
_output_shapes
:?????????@i
dropout_13/dropout/ShapeShape!max_pooling2d_13/MaxPool:output:0*
T0*
_output_shapes
:ͺ
/dropout_13/dropout/random_uniform/RandomUniformRandomUniform!dropout_13/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????@*
dtype0f
!dropout_13/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>Ο
dropout_13/dropout/GreaterEqualGreaterEqual8dropout_13/dropout/random_uniform/RandomUniform:output:0*dropout_13/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????@
dropout_13/dropout/CastCast#dropout_13/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@
dropout_13/dropout/Mul_1Muldropout_13/dropout/Mul:z:0dropout_13/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????@
1global_average_pooling2d_6/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ³
global_average_pooling2d_6/MeanMeandropout_13/dropout/Mul_1:z:0:global_average_pooling2d_6/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????@
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:@0*
dtype0
dense_12/MatMulMatMul(global_average_pooling2d_6/Mean:output:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????0
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????0h
IdentityIdentitydense_12/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????0
NoOpNoOp!^conv2d_12/BiasAdd/ReadVariableOp ^conv2d_12/Conv2D/ReadVariableOp!^conv2d_13/BiasAdd/ReadVariableOp ^conv2d_13/Conv2D/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????dd: : : : : : 2D
 conv2d_12/BiasAdd/ReadVariableOp conv2d_12/BiasAdd/ReadVariableOp2B
conv2d_12/Conv2D/ReadVariableOpconv2d_12/Conv2D/ReadVariableOp2D
 conv2d_13/BiasAdd/ReadVariableOp conv2d_13/BiasAdd/ReadVariableOp2B
conv2d_13/Conv2D/ReadVariableOpconv2d_13/Conv2D/ReadVariableOp2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????dd
 
_user_specified_nameinputs
ξ#
λ
C__inference_model_12_layer_call_and_return_conditional_losses_22675

inputs)
conv2d_12_22654:@
conv2d_12_22656:@)
conv2d_13_22661:@@
conv2d_13_22663:@ 
dense_12_22669:@0
dense_12_22671:0
identity’!conv2d_12/StatefulPartitionedCall’!conv2d_13/StatefulPartitionedCall’ dense_12/StatefulPartitionedCall’"dropout_12/StatefulPartitionedCall’"dropout_13/StatefulPartitionedCallω
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_12_22654conv2d_12_22656*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????dd@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_12_layer_call_and_return_conditional_losses_22484σ
 max_pooling2d_12/PartitionedCallPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????22@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_22438φ
"dropout_12/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_12/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????22@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_12_layer_call_and_return_conditional_losses_22619
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall+dropout_12/StatefulPartitionedCall:output:0conv2d_13_22661conv2d_13_22663*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????22@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_13_layer_call_and_return_conditional_losses_22509σ
 max_pooling2d_13/PartitionedCallPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_max_pooling2d_13_layer_call_and_return_conditional_losses_22450
"dropout_13/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_13/PartitionedCall:output:0#^dropout_12/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_13_layer_call_and_return_conditional_losses_22586
*global_average_pooling2d_6/PartitionedCallPartitionedCall+dropout_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *^
fYRW
U__inference_global_average_pooling2d_6_layer_call_and_return_conditional_losses_22463
 dense_12/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling2d_6/PartitionedCall:output:0dense_12_22669dense_12_22671*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_12_layer_call_and_return_conditional_losses_22534x
IdentityIdentity)dense_12/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????0ϋ
NoOpNoOp"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall#^dropout_12/StatefulPartitionedCall#^dropout_13/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????dd: : : : : : 2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2H
"dropout_12/StatefulPartitionedCall"dropout_12/StatefulPartitionedCall2H
"dropout_13/StatefulPartitionedCall"dropout_13/StatefulPartitionedCall:W S
/
_output_shapes
:?????????dd
 
_user_specified_nameinputs
τ#
ν
C__inference_model_12_layer_call_and_return_conditional_losses_22755
input_21)
conv2d_12_22734:@
conv2d_12_22736:@)
conv2d_13_22741:@@
conv2d_13_22743:@ 
dense_12_22749:@0
dense_12_22751:0
identity’!conv2d_12/StatefulPartitionedCall’!conv2d_13/StatefulPartitionedCall’ dense_12/StatefulPartitionedCall’"dropout_12/StatefulPartitionedCall’"dropout_13/StatefulPartitionedCallϋ
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCallinput_21conv2d_12_22734conv2d_12_22736*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????dd@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_12_layer_call_and_return_conditional_losses_22484σ
 max_pooling2d_12/PartitionedCallPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????22@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_22438φ
"dropout_12/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_12/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????22@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_12_layer_call_and_return_conditional_losses_22619
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall+dropout_12/StatefulPartitionedCall:output:0conv2d_13_22741conv2d_13_22743*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????22@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_13_layer_call_and_return_conditional_losses_22509σ
 max_pooling2d_13/PartitionedCallPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_max_pooling2d_13_layer_call_and_return_conditional_losses_22450
"dropout_13/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_13/PartitionedCall:output:0#^dropout_12/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_13_layer_call_and_return_conditional_losses_22586
*global_average_pooling2d_6/PartitionedCallPartitionedCall+dropout_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *^
fYRW
U__inference_global_average_pooling2d_6_layer_call_and_return_conditional_losses_22463
 dense_12/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling2d_6/PartitionedCall:output:0dense_12_22749dense_12_22751*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_12_layer_call_and_return_conditional_losses_22534x
IdentityIdentity)dense_12/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????0ϋ
NoOpNoOp"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall#^dropout_12/StatefulPartitionedCall#^dropout_13/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????dd: : : : : : 2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2H
"dropout_12/StatefulPartitionedCall"dropout_12/StatefulPartitionedCall2H
"dropout_13/StatefulPartitionedCall"dropout_13/StatefulPartitionedCall:Y U
/
_output_shapes
:?????????dd
"
_user_specified_name
input_21
ψ
c
E__inference_dropout_12_layer_call_and_return_conditional_losses_22496

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????22@c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????22@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????22@:W S
/
_output_shapes
:?????????22@
 
_user_specified_nameinputs
ρ

Χ
(__inference_model_13_layer_call_fn_22837
input_19
input_20!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@
	unknown_3:@0
	unknown_4:0
	unknown_5:
	unknown_6:
identity’StatefulPartitionedCall³
StatefulPartitionedCallStatefulPartitionedCallinput_19input_20unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_model_13_layer_call_and_return_conditional_losses_22818o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:?????????dd:?????????dd: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:?????????dd
"
_user_specified_name
input_19:YU
/
_output_shapes
:?????????dd
"
_user_specified_name
input_20
³

d
E__inference_dropout_13_layer_call_and_return_conditional_losses_22586

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nΫΆ?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????@w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????@a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
Ί$

C__inference_model_12_layer_call_and_return_conditional_losses_23333

inputsB
(conv2d_12_conv2d_readvariableop_resource:@7
)conv2d_12_biasadd_readvariableop_resource:@B
(conv2d_13_conv2d_readvariableop_resource:@@7
)conv2d_13_biasadd_readvariableop_resource:@9
'dense_12_matmul_readvariableop_resource:@06
(dense_12_biasadd_readvariableop_resource:0
identity’ conv2d_12/BiasAdd/ReadVariableOp’conv2d_12/Conv2D/ReadVariableOp’ conv2d_13/BiasAdd/ReadVariableOp’conv2d_13/Conv2D/ReadVariableOp’dense_12/BiasAdd/ReadVariableOp’dense_12/MatMul/ReadVariableOp
conv2d_12/Conv2D/ReadVariableOpReadVariableOp(conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0­
conv2d_12/Conv2DConv2Dinputs'conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????dd@*
paddingSAME*
strides

 conv2d_12/BiasAdd/ReadVariableOpReadVariableOp)conv2d_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_12/BiasAddBiasAddconv2d_12/Conv2D:output:0(conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????dd@l
conv2d_12/ReluReluconv2d_12/BiasAdd:output:0*
T0*/
_output_shapes
:?????????dd@?
max_pooling2d_12/MaxPoolMaxPoolconv2d_12/Relu:activations:0*/
_output_shapes
:?????????22@*
ksize
*
paddingVALID*
strides
|
dropout_12/IdentityIdentity!max_pooling2d_12/MaxPool:output:0*
T0*/
_output_shapes
:?????????22@
conv2d_13/Conv2D/ReadVariableOpReadVariableOp(conv2d_13_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Γ
conv2d_13/Conv2DConv2Ddropout_12/Identity:output:0'conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????22@*
paddingSAME*
strides

 conv2d_13/BiasAdd/ReadVariableOpReadVariableOp)conv2d_13_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_13/BiasAddBiasAddconv2d_13/Conv2D:output:0(conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????22@l
conv2d_13/ReluReluconv2d_13/BiasAdd:output:0*
T0*/
_output_shapes
:?????????22@?
max_pooling2d_13/MaxPoolMaxPoolconv2d_13/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
|
dropout_13/IdentityIdentity!max_pooling2d_13/MaxPool:output:0*
T0*/
_output_shapes
:?????????@
1global_average_pooling2d_6/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ³
global_average_pooling2d_6/MeanMeandropout_13/Identity:output:0:global_average_pooling2d_6/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????@
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:@0*
dtype0
dense_12/MatMulMatMul(global_average_pooling2d_6/Mean:output:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????0
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????0h
IdentityIdentitydense_12/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????0
NoOpNoOp!^conv2d_12/BiasAdd/ReadVariableOp ^conv2d_12/Conv2D/ReadVariableOp!^conv2d_13/BiasAdd/ReadVariableOp ^conv2d_13/Conv2D/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????dd: : : : : : 2D
 conv2d_12/BiasAdd/ReadVariableOp conv2d_12/BiasAdd/ReadVariableOp2B
conv2d_12/Conv2D/ReadVariableOpconv2d_12/Conv2D/ReadVariableOp2D
 conv2d_13/BiasAdd/ReadVariableOp conv2d_13/BiasAdd/ReadVariableOp2B
conv2d_13/Conv2D/ReadVariableOpconv2d_13/Conv2D/ReadVariableOp2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????dd
 
_user_specified_nameinputs
 
T
(__inference_lambda_6_layer_call_fn_23389
inputs_0
inputs_1
identity»
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lambda_6_layer_call_and_return_conditional_losses_22871`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????0:?????????0:Q M
'
_output_shapes
:?????????0
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????0
"
_user_specified_name
inputs/1
λD
Λ
__inference__traced_save_23704
file_prefix.
*savev2_dense_13_kernel_read_readvariableop,
(savev2_dense_13_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop/
+savev2_conv2d_12_kernel_read_readvariableop-
)savev2_conv2d_12_bias_read_readvariableop/
+savev2_conv2d_13_kernel_read_readvariableop-
)savev2_conv2d_13_bias_read_readvariableop.
*savev2_dense_12_kernel_read_readvariableop,
(savev2_dense_12_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop5
1savev2_adam_dense_13_kernel_m_read_readvariableop3
/savev2_adam_dense_13_bias_m_read_readvariableop6
2savev2_adam_conv2d_12_kernel_m_read_readvariableop4
0savev2_adam_conv2d_12_bias_m_read_readvariableop6
2savev2_adam_conv2d_13_kernel_m_read_readvariableop4
0savev2_adam_conv2d_13_bias_m_read_readvariableop5
1savev2_adam_dense_12_kernel_m_read_readvariableop3
/savev2_adam_dense_12_bias_m_read_readvariableop5
1savev2_adam_dense_13_kernel_v_read_readvariableop3
/savev2_adam_dense_13_bias_v_read_readvariableop6
2savev2_adam_conv2d_12_kernel_v_read_readvariableop4
0savev2_adam_conv2d_12_bias_v_read_readvariableop6
2savev2_adam_conv2d_13_kernel_v_read_readvariableop4
0savev2_adam_conv2d_13_bias_v_read_readvariableop5
1savev2_adam_dense_12_kernel_v_read_readvariableop3
/savev2_adam_dense_12_bias_v_read_readvariableop
savev2_const

identity_1’MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ‘
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*Κ
valueΐB½"B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH±
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ­
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_13_kernel_read_readvariableop(savev2_dense_13_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop+savev2_conv2d_12_kernel_read_readvariableop)savev2_conv2d_12_bias_read_readvariableop+savev2_conv2d_13_kernel_read_readvariableop)savev2_conv2d_13_bias_read_readvariableop*savev2_dense_12_kernel_read_readvariableop(savev2_dense_12_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_dense_13_kernel_m_read_readvariableop/savev2_adam_dense_13_bias_m_read_readvariableop2savev2_adam_conv2d_12_kernel_m_read_readvariableop0savev2_adam_conv2d_12_bias_m_read_readvariableop2savev2_adam_conv2d_13_kernel_m_read_readvariableop0savev2_adam_conv2d_13_bias_m_read_readvariableop1savev2_adam_dense_12_kernel_m_read_readvariableop/savev2_adam_dense_12_bias_m_read_readvariableop1savev2_adam_dense_13_kernel_v_read_readvariableop/savev2_adam_dense_13_bias_v_read_readvariableop2savev2_adam_conv2d_12_kernel_v_read_readvariableop0savev2_adam_conv2d_12_bias_v_read_readvariableop2savev2_adam_conv2d_13_kernel_v_read_readvariableop0savev2_adam_conv2d_13_bias_v_read_readvariableop1savev2_adam_dense_12_kernel_v_read_readvariableop/savev2_adam_dense_12_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *0
dtypes&
$2"	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*
_input_shapes
: ::: : : : : :@:@:@@:@:@0:0: : : : :::@:@:@@:@:@0:0:::@:@:@@:@:@0:0: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:@: 	

_output_shapes
:@:,
(
&
_output_shapes
:@@: 

_output_shapes
:@:$ 

_output_shapes

:@0: 

_output_shapes
:0:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:: 

_output_shapes
::,(
&
_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:$ 

_output_shapes

:@0: 

_output_shapes
:0:$ 

_output_shapes

:: 

_output_shapes
::,(
&
_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:$  

_output_shapes

:@0: !

_output_shapes
:0:"

_output_shapes
: 
ρ

Χ
(__inference_model_13_layer_call_fn_23083
inputs_0
inputs_1!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@
	unknown_3:@0
	unknown_4:0
	unknown_5:
	unknown_6:
identity’StatefulPartitionedCall³
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_model_13_layer_call_and_return_conditional_losses_22930o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:?????????dd:?????????dd: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:?????????dd
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:?????????dd
"
_user_specified_name
inputs/1
ΐ

(__inference_dense_12_layer_call_fn_23571

inputs
unknown:@0
	unknown_0:0
identity’StatefulPartitionedCallΨ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_12_layer_call_and_return_conditional_losses_22534o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????0`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs


m
C__inference_lambda_6_layer_call_and_return_conditional_losses_22871

inputs
inputs_1
identityN
subSubinputsinputs_1*
T0*'
_output_shapes
:?????????0K
SquareSquaresub:z:0*
T0*'
_output_shapes
:?????????0W
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :y
SumSum
Square:y:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(N
	Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΏΦ3f
MaximumMaximumSum:output:0Maximum/y:output:0*
T0*'
_output_shapes
:?????????J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    c
	Maximum_1MaximumMaximum:z:0Const:output:0*
T0*'
_output_shapes
:?????????M
SqrtSqrtMaximum_1:z:0*
T0*'
_output_shapes
:?????????P
IdentityIdentitySqrt:y:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????0:?????????0:O K
'
_output_shapes
:?????????0
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????0
 
_user_specified_nameinputs
χ 
£
C__inference_model_12_layer_call_and_return_conditional_losses_22731
input_21)
conv2d_12_22710:@
conv2d_12_22712:@)
conv2d_13_22717:@@
conv2d_13_22719:@ 
dense_12_22725:@0
dense_12_22727:0
identity’!conv2d_12/StatefulPartitionedCall’!conv2d_13/StatefulPartitionedCall’ dense_12/StatefulPartitionedCallϋ
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCallinput_21conv2d_12_22710conv2d_12_22712*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????dd@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_12_layer_call_and_return_conditional_losses_22484σ
 max_pooling2d_12/PartitionedCallPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????22@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_22438ζ
dropout_12/PartitionedCallPartitionedCall)max_pooling2d_12/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????22@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_12_layer_call_and_return_conditional_losses_22496
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall#dropout_12/PartitionedCall:output:0conv2d_13_22717conv2d_13_22719*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????22@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_13_layer_call_and_return_conditional_losses_22509σ
 max_pooling2d_13/PartitionedCallPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_max_pooling2d_13_layer_call_and_return_conditional_losses_22450ζ
dropout_13/PartitionedCallPartitionedCall)max_pooling2d_13/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_13_layer_call_and_return_conditional_losses_22521ψ
*global_average_pooling2d_6/PartitionedCallPartitionedCall#dropout_13/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *^
fYRW
U__inference_global_average_pooling2d_6_layer_call_and_return_conditional_losses_22463
 dense_12/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling2d_6/PartitionedCall:output:0dense_12_22725dense_12_22727*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_12_layer_call_and_return_conditional_losses_22534x
IdentityIdentity)dense_12/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????0±
NoOpNoOp"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????dd: : : : : : 2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall:Y U
/
_output_shapes
:?????????dd
"
_user_specified_name
input_21
Ά
q
U__inference_global_average_pooling2d_6_layer_call_and_return_conditional_losses_23562

inputs
identityg
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs


τ
C__inference_dense_13_layer_call_and_return_conditional_losses_23437

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
λ
C__inference_model_13_layer_call_and_return_conditional_losses_22930

inputs
inputs_1(
model_12_22903:@
model_12_22905:@(
model_12_22907:@@
model_12_22909:@ 
model_12_22911:@0
model_12_22913:0 
dense_13_22924:
dense_13_22926:
identity’ dense_13/StatefulPartitionedCall’ model_12/StatefulPartitionedCall’"model_12/StatefulPartitionedCall_1΅
 model_12/StatefulPartitionedCallStatefulPartitionedCallinputsmodel_12_22903model_12_22905model_12_22907model_12_22909model_12_22911model_12_22913*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????0*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_model_12_layer_call_and_return_conditional_losses_22675Ή
"model_12/StatefulPartitionedCall_1StatefulPartitionedCallinputs_1model_12_22903model_12_22905model_12_22907model_12_22909model_12_22911model_12_22913*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????0*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_model_12_layer_call_and_return_conditional_losses_22675
lambda_6/PartitionedCallPartitionedCall)model_12/StatefulPartitionedCall:output:0+model_12/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lambda_6_layer_call_and_return_conditional_losses_22871
 dense_13/StatefulPartitionedCallStatefulPartitionedCall!lambda_6/PartitionedCall:output:0dense_13_22924dense_13_22926*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_13_layer_call_and_return_conditional_losses_22811x
IdentityIdentity)dense_13/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????±
NoOpNoOp!^dense_13/StatefulPartitionedCall!^model_12/StatefulPartitionedCall#^model_12/StatefulPartitionedCall_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:?????????dd:?????????dd: : : : : : : : 2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 model_12/StatefulPartitionedCall model_12/StatefulPartitionedCall2H
"model_12/StatefulPartitionedCall_1"model_12/StatefulPartitionedCall_1:W S
/
_output_shapes
:?????????dd
 
_user_specified_nameinputs:WS
/
_output_shapes
:?????????dd
 
_user_specified_nameinputs

c
*__inference_dropout_12_layer_call_fn_23477

inputs
identity’StatefulPartitionedCallΘ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????22@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_12_layer_call_and_return_conditional_losses_22619w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????22@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????22@22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????22@
 
_user_specified_nameinputs

ύ
D__inference_conv2d_13_layer_call_and_return_conditional_losses_22509

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity’BiasAdd/ReadVariableOp’Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????22@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????22@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????22@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????22@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????22@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????22@
 
_user_specified_nameinputs


o
C__inference_lambda_6_layer_call_and_return_conditional_losses_23417
inputs_0
inputs_1
identityP
subSubinputs_0inputs_1*
T0*'
_output_shapes
:?????????0K
SquareSquaresub:z:0*
T0*'
_output_shapes
:?????????0W
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :y
SumSum
Square:y:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(N
	Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΏΦ3f
MaximumMaximumSum:output:0Maximum/y:output:0*
T0*'
_output_shapes
:?????????J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    c
	Maximum_1MaximumMaximum:z:0Const:output:0*
T0*'
_output_shapes
:?????????M
SqrtSqrtMaximum_1:z:0*
T0*'
_output_shapes
:?????????P
IdentityIdentitySqrt:y:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????0:?????????0:Q M
'
_output_shapes
:?????????0
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????0
"
_user_specified_name
inputs/1

g
K__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_22438

inputs
identity’
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
Ώ
F
*__inference_dropout_13_layer_call_fn_23529

inputs
identityΈ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_13_layer_call_and_return_conditional_losses_22521h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs

ύ
D__inference_conv2d_13_layer_call_and_return_conditional_losses_23514

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity’BiasAdd/ReadVariableOp’Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????22@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????22@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????22@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????22@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????22@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????22@
 
_user_specified_nameinputs

ύ
D__inference_conv2d_12_layer_call_and_return_conditional_losses_22484

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity’BiasAdd/ReadVariableOp’Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????dd@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????dd@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????dd@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????dd@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????dd: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????dd
 
_user_specified_nameinputs
ψ
c
E__inference_dropout_12_layer_call_and_return_conditional_losses_23482

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????22@c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????22@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????22@:W S
/
_output_shapes
:?????????22@
 
_user_specified_nameinputs
 
T
(__inference_lambda_6_layer_call_fn_23383
inputs_0
inputs_1
identity»
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lambda_6_layer_call_and_return_conditional_losses_22798`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????0:?????????0:Q M
'
_output_shapes
:?????????0
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????0
"
_user_specified_name
inputs/1
Ώ
F
*__inference_dropout_12_layer_call_fn_23472

inputs
identityΈ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????22@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_12_layer_call_and_return_conditional_losses_22496h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????22@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????22@:W S
/
_output_shapes
:?????????22@
 
_user_specified_nameinputs
³

d
E__inference_dropout_12_layer_call_and_return_conditional_losses_22619

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nΫΆ?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????22@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????22@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????22@w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????22@q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????22@a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????22@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????22@:W S
/
_output_shapes
:?????????22@
 
_user_specified_nameinputs
ρ

Χ
(__inference_model_13_layer_call_fn_23061
inputs_0
inputs_1!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@
	unknown_3:@0
	unknown_4:0
	unknown_5:
	unknown_6:
identity’StatefulPartitionedCall³
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_model_13_layer_call_and_return_conditional_losses_22818o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:?????????dd:?????????dd: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:?????????dd
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:?????????dd
"
_user_specified_name
inputs/1
ΐ

(__inference_dense_13_layer_call_fn_23426

inputs
unknown:
	unknown_0:
identity’StatefulPartitionedCallΨ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_13_layer_call_and_return_conditional_losses_22811o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
Έ
L
0__inference_max_pooling2d_12_layer_call_fn_23462

inputs
identityΩ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_22438
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs


τ
C__inference_dense_13_layer_call_and_return_conditional_losses_22811

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
ψ
c
E__inference_dropout_13_layer_call_and_return_conditional_losses_23539

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????@c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs"ΫL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ό
serving_defaultθ
E
input_199
serving_default_input_19:0?????????dd
E
input_209
serving_default_input_20:0?????????dd<
dense_130
StatefulPartitionedCall:0?????????tensorflow/serving/predict:ϋΤ
Ψ
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer_with_weights-1
layer-4
	optimizer
	variables
trainable_variables
	regularization_losses

	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
"
_tf_keras_input_layer
κ
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer-6
layer-7
layer_with_weights-2
layer-8
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_network
₯
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses"
_tf_keras_layer
»

$kernel
%bias
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses"
_tf_keras_layer
σ
,iter

-beta_1

.beta_2
	/decay
0learning_rate$m±%m²1m³2m΄3m΅4mΆ5m·6mΈ$vΉ%vΊ1v»2vΌ3v½4vΎ5vΏ6vΐ"
	optimizer
X
10
21
32
43
54
65
$6
%7"
trackable_list_wrapper
X
10
21
32
43
54
65
$6
%7"
trackable_list_wrapper
 "
trackable_list_wrapper
Κ
7non_trainable_variables

8layers
9metrics
:layer_regularization_losses
;layer_metrics
	variables
trainable_variables
	regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ξ2λ
(__inference_model_13_layer_call_fn_22837
(__inference_model_13_layer_call_fn_23061
(__inference_model_13_layer_call_fn_23083
(__inference_model_13_layer_call_fn_22971ΐ
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
Ϊ2Χ
C__inference_model_13_layer_call_and_return_conditional_losses_23150
C__inference_model_13_layer_call_and_return_conditional_losses_23245
C__inference_model_13_layer_call_and_return_conditional_losses_23002
C__inference_model_13_layer_call_and_return_conditional_losses_23033ΐ
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
ΦBΣ
 __inference__wrapped_model_22429input_19input_20"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
,
<serving_default"
signature_map
"
_tf_keras_input_layer
»

1kernel
2bias
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses"
_tf_keras_layer
₯
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses"
_tf_keras_layer
Ό
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M_random_generator
N__call__
*O&call_and_return_all_conditional_losses"
_tf_keras_layer
»

3kernel
4bias
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses"
_tf_keras_layer
₯
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses"
_tf_keras_layer
Ό
\	variables
]trainable_variables
^regularization_losses
_	keras_api
`_random_generator
a__call__
*b&call_and_return_all_conditional_losses"
_tf_keras_layer
₯
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
g__call__
*h&call_and_return_all_conditional_losses"
_tf_keras_layer
»

5kernel
6bias
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses"
_tf_keras_layer
J
10
21
32
43
54
65"
trackable_list_wrapper
J
10
21
32
43
54
65"
trackable_list_wrapper
 "
trackable_list_wrapper
­
onon_trainable_variables

players
qmetrics
rlayer_regularization_losses
slayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ξ2λ
(__inference_model_12_layer_call_fn_22556
(__inference_model_12_layer_call_fn_23286
(__inference_model_12_layer_call_fn_23303
(__inference_model_12_layer_call_fn_22707ΐ
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
Ϊ2Χ
C__inference_model_12_layer_call_and_return_conditional_losses_23333
C__inference_model_12_layer_call_and_return_conditional_losses_23377
C__inference_model_12_layer_call_and_return_conditional_losses_22731
C__inference_model_12_layer_call_and_return_conditional_losses_22755ΐ
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
tnon_trainable_variables

ulayers
vmetrics
wlayer_regularization_losses
xlayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
2
(__inference_lambda_6_layer_call_fn_23383
(__inference_lambda_6_layer_call_fn_23389ΐ
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
Π2Ν
C__inference_lambda_6_layer_call_and_return_conditional_losses_23403
C__inference_lambda_6_layer_call_and_return_conditional_losses_23417ΐ
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
!:2dense_13/kernel
:2dense_13/bias
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses"
_generic_user_object
?2Ο
(__inference_dense_13_layer_call_fn_23426’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ν2κ
C__inference_dense_13_layer_call_and_return_conditional_losses_23437’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
*:(@2conv2d_12/kernel
:@2conv2d_12/bias
*:(@@2conv2d_13/kernel
:@2conv2d_13/bias
!:@02dense_12/kernel
:02dense_12/bias
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
.
~0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ΣBΠ
#__inference_signature_wrapper_23269input_19input_20"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
.
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
Σ2Π
)__inference_conv2d_12_layer_call_fn_23446’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ξ2λ
D__inference_conv2d_12_layer_call_and_return_conditional_losses_23457’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
Ϊ2Χ
0__inference_max_pooling2d_12_layer_call_fn_23462’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
υ2ς
K__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_23467’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
*__inference_dropout_12_layer_call_fn_23472
*__inference_dropout_12_layer_call_fn_23477΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
Θ2Ε
E__inference_dropout_12_layer_call_and_return_conditional_losses_23482
E__inference_dropout_12_layer_call_and_return_conditional_losses_23494΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
Σ2Π
)__inference_conv2d_13_layer_call_fn_23503’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ξ2λ
D__inference_conv2d_13_layer_call_and_return_conditional_losses_23514’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
Ϊ2Χ
0__inference_max_pooling2d_13_layer_call_fn_23519’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
υ2ς
K__inference_max_pooling2d_13_layer_call_and_return_conditional_losses_23524’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
\	variables
]trainable_variables
^regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
*__inference_dropout_13_layer_call_fn_23529
*__inference_dropout_13_layer_call_fn_23534΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
Θ2Ε
E__inference_dropout_13_layer_call_and_return_conditional_losses_23539
E__inference_dropout_13_layer_call_and_return_conditional_losses_23551΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
 metrics
 ‘layer_regularization_losses
’layer_metrics
c	variables
dtrainable_variables
eregularization_losses
g__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses"
_generic_user_object
δ2α
:__inference_global_average_pooling2d_6_layer_call_fn_23556’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
?2ό
U__inference_global_average_pooling2d_6_layer_call_and_return_conditional_losses_23562’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
.
50
61"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
²
£non_trainable_variables
€layers
₯metrics
 ¦layer_regularization_losses
§layer_metrics
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
?2Ο
(__inference_dense_12_layer_call_fn_23571’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ν2κ
C__inference_dense_12_layer_call_and_return_conditional_losses_23581’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
_
0
1
2
3
4
5
6
7
8"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
R

¨total

©count
ͺ	variables
«	keras_api"
_tf_keras_metric
c

¬total

­count
?
_fn_kwargs
―	variables
°	keras_api"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:  (2total
:  (2count
0
¨0
©1"
trackable_list_wrapper
.
ͺ	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
¬0
­1"
trackable_list_wrapper
.
―	variables"
_generic_user_object
&:$2Adam/dense_13/kernel/m
 :2Adam/dense_13/bias/m
/:-@2Adam/conv2d_12/kernel/m
!:@2Adam/conv2d_12/bias/m
/:-@@2Adam/conv2d_13/kernel/m
!:@2Adam/conv2d_13/bias/m
&:$@02Adam/dense_12/kernel/m
 :02Adam/dense_12/bias/m
&:$2Adam/dense_13/kernel/v
 :2Adam/dense_13/bias/v
/:-@2Adam/conv2d_12/kernel/v
!:@2Adam/conv2d_12/bias/v
/:-@@2Adam/conv2d_13/kernel/v
!:@2Adam/conv2d_13/bias/v
&:$@02Adam/dense_12/kernel/v
 :02Adam/dense_12/bias/vΠ
 __inference__wrapped_model_22429«123456$%j’g
`’]
[X
*'
input_19?????????dd
*'
input_20?????????dd
ͺ "3ͺ0
.
dense_13"
dense_13?????????΄
D__inference_conv2d_12_layer_call_and_return_conditional_losses_23457l127’4
-’*
(%
inputs?????????dd
ͺ "-’*
# 
0?????????dd@
 
)__inference_conv2d_12_layer_call_fn_23446_127’4
-’*
(%
inputs?????????dd
ͺ " ?????????dd@΄
D__inference_conv2d_13_layer_call_and_return_conditional_losses_23514l347’4
-’*
(%
inputs?????????22@
ͺ "-’*
# 
0?????????22@
 
)__inference_conv2d_13_layer_call_fn_23503_347’4
-’*
(%
inputs?????????22@
ͺ " ?????????22@£
C__inference_dense_12_layer_call_and_return_conditional_losses_23581\56/’,
%’"
 
inputs?????????@
ͺ "%’"

0?????????0
 {
(__inference_dense_12_layer_call_fn_23571O56/’,
%’"
 
inputs?????????@
ͺ "?????????0£
C__inference_dense_13_layer_call_and_return_conditional_losses_23437\$%/’,
%’"
 
inputs?????????
ͺ "%’"

0?????????
 {
(__inference_dense_13_layer_call_fn_23426O$%/’,
%’"
 
inputs?????????
ͺ "?????????΅
E__inference_dropout_12_layer_call_and_return_conditional_losses_23482l;’8
1’.
(%
inputs?????????22@
p 
ͺ "-’*
# 
0?????????22@
 ΅
E__inference_dropout_12_layer_call_and_return_conditional_losses_23494l;’8
1’.
(%
inputs?????????22@
p
ͺ "-’*
# 
0?????????22@
 
*__inference_dropout_12_layer_call_fn_23472_;’8
1’.
(%
inputs?????????22@
p 
ͺ " ?????????22@
*__inference_dropout_12_layer_call_fn_23477_;’8
1’.
(%
inputs?????????22@
p
ͺ " ?????????22@΅
E__inference_dropout_13_layer_call_and_return_conditional_losses_23539l;’8
1’.
(%
inputs?????????@
p 
ͺ "-’*
# 
0?????????@
 ΅
E__inference_dropout_13_layer_call_and_return_conditional_losses_23551l;’8
1’.
(%
inputs?????????@
p
ͺ "-’*
# 
0?????????@
 
*__inference_dropout_13_layer_call_fn_23529_;’8
1’.
(%
inputs?????????@
p 
ͺ " ?????????@
*__inference_dropout_13_layer_call_fn_23534_;’8
1’.
(%
inputs?????????@
p
ͺ " ?????????@ή
U__inference_global_average_pooling2d_6_layer_call_and_return_conditional_losses_23562R’O
H’E
C@
inputs4????????????????????????????????????
ͺ ".’+
$!
0??????????????????
 ΅
:__inference_global_average_pooling2d_6_layer_call_fn_23556wR’O
H’E
C@
inputs4????????????????????????????????????
ͺ "!??????????????????Σ
C__inference_lambda_6_layer_call_and_return_conditional_losses_23403b’_
X’U
KH
"
inputs/0?????????0
"
inputs/1?????????0

 
p 
ͺ "%’"

0?????????
 Σ
C__inference_lambda_6_layer_call_and_return_conditional_losses_23417b’_
X’U
KH
"
inputs/0?????????0
"
inputs/1?????????0

 
p
ͺ "%’"

0?????????
 ͺ
(__inference_lambda_6_layer_call_fn_23383~b’_
X’U
KH
"
inputs/0?????????0
"
inputs/1?????????0

 
p 
ͺ "?????????ͺ
(__inference_lambda_6_layer_call_fn_23389~b’_
X’U
KH
"
inputs/0?????????0
"
inputs/1?????????0

 
p
ͺ "?????????ξ
K__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_23467R’O
H’E
C@
inputs4????????????????????????????????????
ͺ "H’E
>;
04????????????????????????????????????
 Ζ
0__inference_max_pooling2d_12_layer_call_fn_23462R’O
H’E
C@
inputs4????????????????????????????????????
ͺ ";84????????????????????????????????????ξ
K__inference_max_pooling2d_13_layer_call_and_return_conditional_losses_23524R’O
H’E
C@
inputs4????????????????????????????????????
ͺ "H’E
>;
04????????????????????????????????????
 Ζ
0__inference_max_pooling2d_13_layer_call_fn_23519R’O
H’E
C@
inputs4????????????????????????????????????
ͺ ";84????????????????????????????????????Ή
C__inference_model_12_layer_call_and_return_conditional_losses_22731r123456A’>
7’4
*'
input_21?????????dd
p 

 
ͺ "%’"

0?????????0
 Ή
C__inference_model_12_layer_call_and_return_conditional_losses_22755r123456A’>
7’4
*'
input_21?????????dd
p

 
ͺ "%’"

0?????????0
 ·
C__inference_model_12_layer_call_and_return_conditional_losses_23333p123456?’<
5’2
(%
inputs?????????dd
p 

 
ͺ "%’"

0?????????0
 ·
C__inference_model_12_layer_call_and_return_conditional_losses_23377p123456?’<
5’2
(%
inputs?????????dd
p

 
ͺ "%’"

0?????????0
 
(__inference_model_12_layer_call_fn_22556e123456A’>
7’4
*'
input_21?????????dd
p 

 
ͺ "?????????0
(__inference_model_12_layer_call_fn_22707e123456A’>
7’4
*'
input_21?????????dd
p

 
ͺ "?????????0
(__inference_model_12_layer_call_fn_23286c123456?’<
5’2
(%
inputs?????????dd
p 

 
ͺ "?????????0
(__inference_model_12_layer_call_fn_23303c123456?’<
5’2
(%
inputs?????????dd
p

 
ͺ "?????????0ν
C__inference_model_13_layer_call_and_return_conditional_losses_23002₯123456$%r’o
h’e
[X
*'
input_19?????????dd
*'
input_20?????????dd
p 

 
ͺ "%’"

0?????????
 ν
C__inference_model_13_layer_call_and_return_conditional_losses_23033₯123456$%r’o
h’e
[X
*'
input_19?????????dd
*'
input_20?????????dd
p

 
ͺ "%’"

0?????????
 ν
C__inference_model_13_layer_call_and_return_conditional_losses_23150₯123456$%r’o
h’e
[X
*'
inputs/0?????????dd
*'
inputs/1?????????dd
p 

 
ͺ "%’"

0?????????
 ν
C__inference_model_13_layer_call_and_return_conditional_losses_23245₯123456$%r’o
h’e
[X
*'
inputs/0?????????dd
*'
inputs/1?????????dd
p

 
ͺ "%’"

0?????????
 Ε
(__inference_model_13_layer_call_fn_22837123456$%r’o
h’e
[X
*'
input_19?????????dd
*'
input_20?????????dd
p 

 
ͺ "?????????Ε
(__inference_model_13_layer_call_fn_22971123456$%r’o
h’e
[X
*'
input_19?????????dd
*'
input_20?????????dd
p

 
ͺ "?????????Ε
(__inference_model_13_layer_call_fn_23061123456$%r’o
h’e
[X
*'
inputs/0?????????dd
*'
inputs/1?????????dd
p 

 
ͺ "?????????Ε
(__inference_model_13_layer_call_fn_23083123456$%r’o
h’e
[X
*'
inputs/0?????????dd
*'
inputs/1?????????dd
p

 
ͺ "?????????ζ
#__inference_signature_wrapper_23269Ύ123456$%}’z
’ 
sͺp
6
input_19*'
input_19?????????dd
6
input_20*'
input_20?????????dd"3ͺ0
.
dense_13"
dense_13?????????