��
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
8
Const
output"dtype"
valuetensor"
dtypetype
�
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
�
DepthwiseConv2dNative

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
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
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
?
Mul
x"T
y"T
z"T"
Ttype:
2	�
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
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
@
ReadVariableOp
resource
value"dtype"
dtypetype�
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
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
list(type)(0�
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
A
SelectV2
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	�
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
�
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
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
2
StopGradient

input"T
output"T"	
Ttype
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
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.11.02v2.11.0-rc2-15-g6290819256d8�
d
VariableVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Variable
]
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
: *
dtype0
h

Variable_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Variable_1
a
Variable_1/Read/ReadVariableOpReadVariableOp
Variable_1*
_output_shapes
: *
dtype0
h

Variable_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Variable_2
a
Variable_2/Read/ReadVariableOpReadVariableOp
Variable_2*
_output_shapes
: *
dtype0
h

Variable_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Variable_3
a
Variable_3/Read/ReadVariableOpReadVariableOp
Variable_3*
_output_shapes
: *
dtype0
h

Variable_4VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Variable_4
a
Variable_4/Read/ReadVariableOpReadVariableOp
Variable_4*
_output_shapes
: *
dtype0
h

Variable_5VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Variable_5
a
Variable_5/Read/ReadVariableOpReadVariableOp
Variable_5*
_output_shapes
: *
dtype0
h

Variable_6VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Variable_6
a
Variable_6/Read/ReadVariableOpReadVariableOp
Variable_6*
_output_shapes
: *
dtype0
h

Variable_7VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Variable_7
a
Variable_7/Read/ReadVariableOpReadVariableOp
Variable_7*
_output_shapes
: *
dtype0
h

Variable_8VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Variable_8
a
Variable_8/Read/ReadVariableOpReadVariableOp
Variable_8*
_output_shapes
: *
dtype0
h

Variable_9VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Variable_9
a
Variable_9/Read/ReadVariableOpReadVariableOp
Variable_9*
_output_shapes
: *
dtype0
j
Variable_10VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameVariable_10
c
Variable_10/Read/ReadVariableOpReadVariableOpVariable_10*
_output_shapes
: *
dtype0
j
Variable_11VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameVariable_11
c
Variable_11/Read/ReadVariableOpReadVariableOpVariable_11*
_output_shapes
: *
dtype0
j
Variable_12VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameVariable_12
c
Variable_12/Read/ReadVariableOpReadVariableOpVariable_12*
_output_shapes
: *
dtype0
j
Variable_13VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameVariable_13
c
Variable_13/Read/ReadVariableOpReadVariableOpVariable_13*
_output_shapes
: *
dtype0
j
Variable_14VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameVariable_14
c
Variable_14/Read/ReadVariableOpReadVariableOpVariable_14*
_output_shapes
: *
dtype0
j
Variable_15VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameVariable_15
c
Variable_15/Read/ReadVariableOpReadVariableOpVariable_15*
_output_shapes
: *
dtype0
j
Variable_16VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameVariable_16
c
Variable_16/Read/ReadVariableOpReadVariableOpVariable_16*
_output_shapes
: *
dtype0
j
Variable_17VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameVariable_17
c
Variable_17/Read/ReadVariableOpReadVariableOpVariable_17*
_output_shapes
: *
dtype0
j
Variable_18VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameVariable_18
c
Variable_18/Read/ReadVariableOpReadVariableOpVariable_18*
_output_shapes
: *
dtype0
|
Variable_19VarHandleOp*
_output_shapes
: *
dtype0*
shape:��*
shared_nameVariable_19
u
Variable_19/Read/ReadVariableOpReadVariableOpVariable_19*(
_output_shapes
:��*
dtype0
{
Variable_20VarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameVariable_20
t
Variable_20/Read/ReadVariableOpReadVariableOpVariable_20*'
_output_shapes
:�*
dtype0
j
Variable_21VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameVariable_21
c
Variable_21/Read/ReadVariableOpReadVariableOpVariable_21*
_output_shapes
: *
dtype0
|
Variable_22VarHandleOp*
_output_shapes
: *
dtype0*
shape:��*
shared_nameVariable_22
u
Variable_22/Read/ReadVariableOpReadVariableOpVariable_22*(
_output_shapes
:��*
dtype0
{
Variable_23VarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameVariable_23
t
Variable_23/Read/ReadVariableOpReadVariableOpVariable_23*'
_output_shapes
:�*
dtype0
j
Variable_24VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameVariable_24
c
Variable_24/Read/ReadVariableOpReadVariableOpVariable_24*
_output_shapes
: *
dtype0
{
Variable_25VarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*
shared_nameVariable_25
t
Variable_25/Read/ReadVariableOpReadVariableOpVariable_25*'
_output_shapes
:@�*
dtype0
z
Variable_26VarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameVariable_26
s
Variable_26/Read/ReadVariableOpReadVariableOpVariable_26*&
_output_shapes
:@*
dtype0
j
Variable_27VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameVariable_27
c
Variable_27/Read/ReadVariableOpReadVariableOpVariable_27*
_output_shapes
: *
dtype0
z
Variable_28VarHandleOp*
_output_shapes
: *
dtype0*
shape: @*
shared_nameVariable_28
s
Variable_28/Read/ReadVariableOpReadVariableOpVariable_28*&
_output_shapes
: @*
dtype0
z
Variable_29VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameVariable_29
s
Variable_29/Read/ReadVariableOpReadVariableOpVariable_29*&
_output_shapes
: *
dtype0
j
Variable_30VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameVariable_30
c
Variable_30/Read/ReadVariableOpReadVariableOpVariable_30*
_output_shapes
: *
dtype0
z
Variable_31VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameVariable_31
s
Variable_31/Read/ReadVariableOpReadVariableOpVariable_31*&
_output_shapes
: *
dtype0
n
Variable_32VarHandleOp*
_output_shapes
: *
dtype0*
shape:e*
shared_nameVariable_32
g
Variable_32/Read/ReadVariableOpReadVariableOpVariable_32*
_output_shapes
:e*
dtype0
s
Variable_33VarHandleOp*
_output_shapes
: *
dtype0*
shape:	�e*
shared_nameVariable_33
l
Variable_33/Read/ReadVariableOpReadVariableOpVariable_33*
_output_shapes
:	�e*
dtype0
o
Variable_34VarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameVariable_34
h
Variable_34/Read/ReadVariableOpReadVariableOpVariable_34*
_output_shapes	
:�*
dtype0
t
Variable_35VarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_nameVariable_35
m
Variable_35/Read/ReadVariableOpReadVariableOpVariable_35* 
_output_shapes
:
��*
dtype0
�
serving_default_inputsPlaceholder*1
_output_shapes
:�����������*
dtype0*&
shape:�����������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_inputsVariable_31Variable_17Variable_16Variable_30Variable_29Variable_15Variable_14Variable_28Variable_13Variable_12Variable_27Variable_26Variable_11Variable_10Variable_25
Variable_9
Variable_8Variable_24Variable_23
Variable_7
Variable_6Variable_22
Variable_5
Variable_4Variable_21Variable_20
Variable_3
Variable_2Variable_19
Variable_1VariableVariable_18Variable_35Variable_34Variable_33Variable_32*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������e*F
_read_only_resource_inputs(
&$	
 !"#$*-
config_proto

CPU

GPU 2J 8� */
f*R(
&__inference_signature_wrapper_52924652

NoOpNoOp
�"
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�"
value�"B�" B�!
�
conv_block1
depthwise_conv_block1
depthwise_conv_block2
depthwise_conv_block3
depthwise_conv_block4
dense_weights1
dense_bias1
dense_weights2
	dense_bias2

serving_default

signatures*
+
conv_weights
bn
dropout*
O
depthwise_filter
pointwise_filter
bn1
bn2
dropout*
O
depthwise_filter
pointwise_filter
bn1
bn2
dropout*
O
depthwise_filter
pointwise_filter
bn1
bn2
dropout*
O
depthwise_filter
pointwise_filter
 bn1
!bn2
"dropout*
NH
VARIABLE_VALUEVariable_35)dense_weights1/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEVariable_34&dense_bias1/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEVariable_33)dense_weights2/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEVariable_32&dense_bias2/.ATTRIBUTES/VARIABLE_VALUE*

#trace_0* 

$serving_default* 
XR
VARIABLE_VALUEVariable_313conv_block1/conv_weights/.ATTRIBUTES/VARIABLE_VALUE*

	%scale

&offset*
SM
VARIABLE_VALUEVariable_30.conv_block1/dropout/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEVariable_29Adepthwise_conv_block1/depthwise_filter/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEVariable_28Adepthwise_conv_block1/pointwise_filter/.ATTRIBUTES/VARIABLE_VALUE*

	'scale

(offset*

	)scale

*offset*
]W
VARIABLE_VALUEVariable_278depthwise_conv_block1/dropout/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEVariable_26Adepthwise_conv_block2/depthwise_filter/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEVariable_25Adepthwise_conv_block2/pointwise_filter/.ATTRIBUTES/VARIABLE_VALUE*

	+scale

,offset*

	-scale

.offset*
]W
VARIABLE_VALUEVariable_248depthwise_conv_block2/dropout/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEVariable_23Adepthwise_conv_block3/depthwise_filter/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEVariable_22Adepthwise_conv_block3/pointwise_filter/.ATTRIBUTES/VARIABLE_VALUE*

	/scale

0offset*

	1scale

2offset*
]W
VARIABLE_VALUEVariable_218depthwise_conv_block3/dropout/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEVariable_20Adepthwise_conv_block4/depthwise_filter/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEVariable_19Adepthwise_conv_block4/pointwise_filter/.ATTRIBUTES/VARIABLE_VALUE*

	3scale

4offset*

	5scale

6offset*
]W
VARIABLE_VALUEVariable_188depthwise_conv_block4/dropout/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
TN
VARIABLE_VALUEVariable_17/conv_block1/bn/scale/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEVariable_160conv_block1/bn/offset/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEVariable_15:depthwise_conv_block1/bn1/scale/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEVariable_14;depthwise_conv_block1/bn1/offset/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEVariable_13:depthwise_conv_block1/bn2/scale/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEVariable_12;depthwise_conv_block1/bn2/offset/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEVariable_11:depthwise_conv_block2/bn1/scale/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEVariable_10;depthwise_conv_block2/bn1/offset/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUE
Variable_9:depthwise_conv_block2/bn2/scale/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUE
Variable_8;depthwise_conv_block2/bn2/offset/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUE
Variable_7:depthwise_conv_block3/bn1/scale/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUE
Variable_6;depthwise_conv_block3/bn1/offset/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUE
Variable_5:depthwise_conv_block3/bn2/scale/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUE
Variable_4;depthwise_conv_block3/bn2/offset/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUE
Variable_3:depthwise_conv_block4/bn1/scale/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUE
Variable_2;depthwise_conv_block4/bn1/offset/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUE
Variable_1:depthwise_conv_block4/bn2/scale/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEVariable;depthwise_conv_block4/bn2/offset/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameVariable_35/Read/ReadVariableOpVariable_34/Read/ReadVariableOpVariable_33/Read/ReadVariableOpVariable_32/Read/ReadVariableOpVariable_31/Read/ReadVariableOpVariable_30/Read/ReadVariableOpVariable_29/Read/ReadVariableOpVariable_28/Read/ReadVariableOpVariable_27/Read/ReadVariableOpVariable_26/Read/ReadVariableOpVariable_25/Read/ReadVariableOpVariable_24/Read/ReadVariableOpVariable_23/Read/ReadVariableOpVariable_22/Read/ReadVariableOpVariable_21/Read/ReadVariableOpVariable_20/Read/ReadVariableOpVariable_19/Read/ReadVariableOpVariable_18/Read/ReadVariableOpVariable_17/Read/ReadVariableOpVariable_16/Read/ReadVariableOpVariable_15/Read/ReadVariableOpVariable_14/Read/ReadVariableOpVariable_13/Read/ReadVariableOpVariable_12/Read/ReadVariableOpVariable_11/Read/ReadVariableOpVariable_10/Read/ReadVariableOpVariable_9/Read/ReadVariableOpVariable_8/Read/ReadVariableOpVariable_7/Read/ReadVariableOpVariable_6/Read/ReadVariableOpVariable_5/Read/ReadVariableOpVariable_4/Read/ReadVariableOpVariable_3/Read/ReadVariableOpVariable_2/Read/ReadVariableOpVariable_1/Read/ReadVariableOpVariable/Read/ReadVariableOpConst*1
Tin*
(2&*
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
GPU 2J 8� **
f%R#
!__inference__traced_save_52924783
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameVariable_35Variable_34Variable_33Variable_32Variable_31Variable_30Variable_29Variable_28Variable_27Variable_26Variable_25Variable_24Variable_23Variable_22Variable_21Variable_20Variable_19Variable_18Variable_17Variable_16Variable_15Variable_14Variable_13Variable_12Variable_11Variable_10
Variable_9
Variable_8
Variable_7
Variable_6
Variable_5
Variable_4
Variable_3
Variable_2
Variable_1Variable*0
Tin)
'2%*
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
GPU 2J 8� *-
f(R&
$__inference__traced_restore_52924901֦
�F
�
!__inference__traced_save_52924783
file_prefix*
&savev2_variable_35_read_readvariableop*
&savev2_variable_34_read_readvariableop*
&savev2_variable_33_read_readvariableop*
&savev2_variable_32_read_readvariableop*
&savev2_variable_31_read_readvariableop*
&savev2_variable_30_read_readvariableop*
&savev2_variable_29_read_readvariableop*
&savev2_variable_28_read_readvariableop*
&savev2_variable_27_read_readvariableop*
&savev2_variable_26_read_readvariableop*
&savev2_variable_25_read_readvariableop*
&savev2_variable_24_read_readvariableop*
&savev2_variable_23_read_readvariableop*
&savev2_variable_22_read_readvariableop*
&savev2_variable_21_read_readvariableop*
&savev2_variable_20_read_readvariableop*
&savev2_variable_19_read_readvariableop*
&savev2_variable_18_read_readvariableop*
&savev2_variable_17_read_readvariableop*
&savev2_variable_16_read_readvariableop*
&savev2_variable_15_read_readvariableop*
&savev2_variable_14_read_readvariableop*
&savev2_variable_13_read_readvariableop*
&savev2_variable_12_read_readvariableop*
&savev2_variable_11_read_readvariableop*
&savev2_variable_10_read_readvariableop)
%savev2_variable_9_read_readvariableop)
%savev2_variable_8_read_readvariableop)
%savev2_variable_7_read_readvariableop)
%savev2_variable_6_read_readvariableop)
%savev2_variable_5_read_readvariableop)
%savev2_variable_4_read_readvariableop)
%savev2_variable_3_read_readvariableop)
%savev2_variable_2_read_readvariableop)
%savev2_variable_1_read_readvariableop'
#savev2_variable_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
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
_temp/part�
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
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*�
value�B�%B)dense_weights1/.ATTRIBUTES/VARIABLE_VALUEB&dense_bias1/.ATTRIBUTES/VARIABLE_VALUEB)dense_weights2/.ATTRIBUTES/VARIABLE_VALUEB&dense_bias2/.ATTRIBUTES/VARIABLE_VALUEB3conv_block1/conv_weights/.ATTRIBUTES/VARIABLE_VALUEB.conv_block1/dropout/.ATTRIBUTES/VARIABLE_VALUEBAdepthwise_conv_block1/depthwise_filter/.ATTRIBUTES/VARIABLE_VALUEBAdepthwise_conv_block1/pointwise_filter/.ATTRIBUTES/VARIABLE_VALUEB8depthwise_conv_block1/dropout/.ATTRIBUTES/VARIABLE_VALUEBAdepthwise_conv_block2/depthwise_filter/.ATTRIBUTES/VARIABLE_VALUEBAdepthwise_conv_block2/pointwise_filter/.ATTRIBUTES/VARIABLE_VALUEB8depthwise_conv_block2/dropout/.ATTRIBUTES/VARIABLE_VALUEBAdepthwise_conv_block3/depthwise_filter/.ATTRIBUTES/VARIABLE_VALUEBAdepthwise_conv_block3/pointwise_filter/.ATTRIBUTES/VARIABLE_VALUEB8depthwise_conv_block3/dropout/.ATTRIBUTES/VARIABLE_VALUEBAdepthwise_conv_block4/depthwise_filter/.ATTRIBUTES/VARIABLE_VALUEBAdepthwise_conv_block4/pointwise_filter/.ATTRIBUTES/VARIABLE_VALUEB8depthwise_conv_block4/dropout/.ATTRIBUTES/VARIABLE_VALUEB/conv_block1/bn/scale/.ATTRIBUTES/VARIABLE_VALUEB0conv_block1/bn/offset/.ATTRIBUTES/VARIABLE_VALUEB:depthwise_conv_block1/bn1/scale/.ATTRIBUTES/VARIABLE_VALUEB;depthwise_conv_block1/bn1/offset/.ATTRIBUTES/VARIABLE_VALUEB:depthwise_conv_block1/bn2/scale/.ATTRIBUTES/VARIABLE_VALUEB;depthwise_conv_block1/bn2/offset/.ATTRIBUTES/VARIABLE_VALUEB:depthwise_conv_block2/bn1/scale/.ATTRIBUTES/VARIABLE_VALUEB;depthwise_conv_block2/bn1/offset/.ATTRIBUTES/VARIABLE_VALUEB:depthwise_conv_block2/bn2/scale/.ATTRIBUTES/VARIABLE_VALUEB;depthwise_conv_block2/bn2/offset/.ATTRIBUTES/VARIABLE_VALUEB:depthwise_conv_block3/bn1/scale/.ATTRIBUTES/VARIABLE_VALUEB;depthwise_conv_block3/bn1/offset/.ATTRIBUTES/VARIABLE_VALUEB:depthwise_conv_block3/bn2/scale/.ATTRIBUTES/VARIABLE_VALUEB;depthwise_conv_block3/bn2/offset/.ATTRIBUTES/VARIABLE_VALUEB:depthwise_conv_block4/bn1/scale/.ATTRIBUTES/VARIABLE_VALUEB;depthwise_conv_block4/bn1/offset/.ATTRIBUTES/VARIABLE_VALUEB:depthwise_conv_block4/bn2/scale/.ATTRIBUTES/VARIABLE_VALUEB;depthwise_conv_block4/bn2/offset/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*]
valueTBR%B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0&savev2_variable_35_read_readvariableop&savev2_variable_34_read_readvariableop&savev2_variable_33_read_readvariableop&savev2_variable_32_read_readvariableop&savev2_variable_31_read_readvariableop&savev2_variable_30_read_readvariableop&savev2_variable_29_read_readvariableop&savev2_variable_28_read_readvariableop&savev2_variable_27_read_readvariableop&savev2_variable_26_read_readvariableop&savev2_variable_25_read_readvariableop&savev2_variable_24_read_readvariableop&savev2_variable_23_read_readvariableop&savev2_variable_22_read_readvariableop&savev2_variable_21_read_readvariableop&savev2_variable_20_read_readvariableop&savev2_variable_19_read_readvariableop&savev2_variable_18_read_readvariableop&savev2_variable_17_read_readvariableop&savev2_variable_16_read_readvariableop&savev2_variable_15_read_readvariableop&savev2_variable_14_read_readvariableop&savev2_variable_13_read_readvariableop&savev2_variable_12_read_readvariableop&savev2_variable_11_read_readvariableop&savev2_variable_10_read_readvariableop%savev2_variable_9_read_readvariableop%savev2_variable_8_read_readvariableop%savev2_variable_7_read_readvariableop%savev2_variable_6_read_readvariableop%savev2_variable_5_read_readvariableop%savev2_variable_4_read_readvariableop%savev2_variable_3_read_readvariableop%savev2_variable_2_read_readvariableop%savev2_variable_1_read_readvariableop#savev2_variable_read_readvariableopsavev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *3
dtypes)
'2%�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
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

identity_1Identity_1:output:0*�
_input_shapes�
�: :
��:�:	�e:e: : : : @: :@:@�: :�:��: :�:��: : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:%!

_output_shapes
:	�e: 

_output_shapes
:e:,(
&
_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: :,(
&
_output_shapes
: @:	

_output_shapes
: :,
(
&
_output_shapes
:@:-)
'
_output_shapes
:@�:

_output_shapes
: :-)
'
_output_shapes
:�:.*
(
_output_shapes
:��:

_output_shapes
: :-)
'
_output_shapes
:�:.*
(
_output_shapes
:��:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: 
�
�
$__inference__traced_restore_52924901
file_prefix0
assignvariableop_variable_35:
��-
assignvariableop_1_variable_34:	�1
assignvariableop_2_variable_33:	�e,
assignvariableop_3_variable_32:e8
assignvariableop_4_variable_31: (
assignvariableop_5_variable_30: 8
assignvariableop_6_variable_29: 8
assignvariableop_7_variable_28: @(
assignvariableop_8_variable_27: 8
assignvariableop_9_variable_26:@:
assignvariableop_10_variable_25:@�)
assignvariableop_11_variable_24: :
assignvariableop_12_variable_23:�;
assignvariableop_13_variable_22:��)
assignvariableop_14_variable_21: :
assignvariableop_15_variable_20:�;
assignvariableop_16_variable_19:��)
assignvariableop_17_variable_18: )
assignvariableop_18_variable_17: )
assignvariableop_19_variable_16: )
assignvariableop_20_variable_15: )
assignvariableop_21_variable_14: )
assignvariableop_22_variable_13: )
assignvariableop_23_variable_12: )
assignvariableop_24_variable_11: )
assignvariableop_25_variable_10: (
assignvariableop_26_variable_9: (
assignvariableop_27_variable_8: (
assignvariableop_28_variable_7: (
assignvariableop_29_variable_6: (
assignvariableop_30_variable_5: (
assignvariableop_31_variable_4: (
assignvariableop_32_variable_3: (
assignvariableop_33_variable_2: (
assignvariableop_34_variable_1: &
assignvariableop_35_variable: 
identity_37��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*�
value�B�%B)dense_weights1/.ATTRIBUTES/VARIABLE_VALUEB&dense_bias1/.ATTRIBUTES/VARIABLE_VALUEB)dense_weights2/.ATTRIBUTES/VARIABLE_VALUEB&dense_bias2/.ATTRIBUTES/VARIABLE_VALUEB3conv_block1/conv_weights/.ATTRIBUTES/VARIABLE_VALUEB.conv_block1/dropout/.ATTRIBUTES/VARIABLE_VALUEBAdepthwise_conv_block1/depthwise_filter/.ATTRIBUTES/VARIABLE_VALUEBAdepthwise_conv_block1/pointwise_filter/.ATTRIBUTES/VARIABLE_VALUEB8depthwise_conv_block1/dropout/.ATTRIBUTES/VARIABLE_VALUEBAdepthwise_conv_block2/depthwise_filter/.ATTRIBUTES/VARIABLE_VALUEBAdepthwise_conv_block2/pointwise_filter/.ATTRIBUTES/VARIABLE_VALUEB8depthwise_conv_block2/dropout/.ATTRIBUTES/VARIABLE_VALUEBAdepthwise_conv_block3/depthwise_filter/.ATTRIBUTES/VARIABLE_VALUEBAdepthwise_conv_block3/pointwise_filter/.ATTRIBUTES/VARIABLE_VALUEB8depthwise_conv_block3/dropout/.ATTRIBUTES/VARIABLE_VALUEBAdepthwise_conv_block4/depthwise_filter/.ATTRIBUTES/VARIABLE_VALUEBAdepthwise_conv_block4/pointwise_filter/.ATTRIBUTES/VARIABLE_VALUEB8depthwise_conv_block4/dropout/.ATTRIBUTES/VARIABLE_VALUEB/conv_block1/bn/scale/.ATTRIBUTES/VARIABLE_VALUEB0conv_block1/bn/offset/.ATTRIBUTES/VARIABLE_VALUEB:depthwise_conv_block1/bn1/scale/.ATTRIBUTES/VARIABLE_VALUEB;depthwise_conv_block1/bn1/offset/.ATTRIBUTES/VARIABLE_VALUEB:depthwise_conv_block1/bn2/scale/.ATTRIBUTES/VARIABLE_VALUEB;depthwise_conv_block1/bn2/offset/.ATTRIBUTES/VARIABLE_VALUEB:depthwise_conv_block2/bn1/scale/.ATTRIBUTES/VARIABLE_VALUEB;depthwise_conv_block2/bn1/offset/.ATTRIBUTES/VARIABLE_VALUEB:depthwise_conv_block2/bn2/scale/.ATTRIBUTES/VARIABLE_VALUEB;depthwise_conv_block2/bn2/offset/.ATTRIBUTES/VARIABLE_VALUEB:depthwise_conv_block3/bn1/scale/.ATTRIBUTES/VARIABLE_VALUEB;depthwise_conv_block3/bn1/offset/.ATTRIBUTES/VARIABLE_VALUEB:depthwise_conv_block3/bn2/scale/.ATTRIBUTES/VARIABLE_VALUEB;depthwise_conv_block3/bn2/offset/.ATTRIBUTES/VARIABLE_VALUEB:depthwise_conv_block4/bn1/scale/.ATTRIBUTES/VARIABLE_VALUEB;depthwise_conv_block4/bn1/offset/.ATTRIBUTES/VARIABLE_VALUEB:depthwise_conv_block4/bn2/scale/.ATTRIBUTES/VARIABLE_VALUEB;depthwise_conv_block4/bn2/offset/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*]
valueTBR%B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::*3
dtypes)
'2%[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_variable_35Identity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_variable_34Identity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOpassignvariableop_2_variable_33Identity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpassignvariableop_3_variable_32Identity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOpassignvariableop_4_variable_31Identity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_variable_30Identity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOpassignvariableop_6_variable_29Identity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpassignvariableop_7_variable_28Identity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOpassignvariableop_8_variable_27Identity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOpassignvariableop_9_variable_26Identity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOpassignvariableop_10_variable_25Identity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOpassignvariableop_11_variable_24Identity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOpassignvariableop_12_variable_23Identity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOpassignvariableop_13_variable_22Identity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOpassignvariableop_14_variable_21Identity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOpassignvariableop_15_variable_20Identity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOpassignvariableop_16_variable_19Identity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOpassignvariableop_17_variable_18Identity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOpassignvariableop_18_variable_17Identity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOpassignvariableop_19_variable_16Identity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOpassignvariableop_20_variable_15Identity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOpassignvariableop_21_variable_14Identity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOpassignvariableop_22_variable_13Identity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOpassignvariableop_23_variable_12Identity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOpassignvariableop_24_variable_11Identity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOpassignvariableop_25_variable_10Identity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOpassignvariableop_26_variable_9Identity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOpassignvariableop_27_variable_8Identity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOpassignvariableop_28_variable_7Identity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOpassignvariableop_29_variable_6Identity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOpassignvariableop_30_variable_5Identity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOpassignvariableop_31_variable_4Identity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOpassignvariableop_32_variable_3Identity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOpassignvariableop_33_variable_2Identity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOpassignvariableop_34_variable_1Identity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOpassignvariableop_35_variableIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_36Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_37IdentityIdentity_36:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_37Identity_37:output:0*]
_input_shapesL
J: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352(
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
�
�
$__inference_serving_default_52924573

inputs8
conv2d_readvariableop_resource: /
%batchnorm_mul_readvariableop_resource: +
!batchnorm_readvariableop_resource: -
#dropout_sub_readvariableop_resource: ;
!depthwise_readvariableop_resource: 1
'batchnorm_1_mul_readvariableop_resource: -
#batchnorm_1_readvariableop_resource: :
 conv2d_1_readvariableop_resource: @1
'batchnorm_2_mul_readvariableop_resource: -
#batchnorm_2_readvariableop_resource: /
%dropout_1_sub_readvariableop_resource: =
#depthwise_1_readvariableop_resource:@1
'batchnorm_3_mul_readvariableop_resource: -
#batchnorm_3_readvariableop_resource: ;
 conv2d_2_readvariableop_resource:@�1
'batchnorm_4_mul_readvariableop_resource: -
#batchnorm_4_readvariableop_resource: /
%dropout_2_sub_readvariableop_resource: >
#depthwise_2_readvariableop_resource:�1
'batchnorm_5_mul_readvariableop_resource: -
#batchnorm_5_readvariableop_resource: <
 conv2d_3_readvariableop_resource:��1
'batchnorm_6_mul_readvariableop_resource: -
#batchnorm_6_readvariableop_resource: /
%dropout_3_sub_readvariableop_resource: >
#depthwise_3_readvariableop_resource:�1
'batchnorm_7_mul_readvariableop_resource: -
#batchnorm_7_readvariableop_resource: <
 conv2d_4_readvariableop_resource:��1
'batchnorm_8_mul_readvariableop_resource: -
#batchnorm_8_readvariableop_resource: /
%dropout_4_sub_readvariableop_resource: 2
matmul_readvariableop_resource:
��*
add_readvariableop_resource:	�3
 matmul_1_readvariableop_resource:	�e+
add_1_readvariableop_resource:e
identity��Conv2D/ReadVariableOp�Conv2D_1/ReadVariableOp�Conv2D_2/ReadVariableOp�Conv2D_3/ReadVariableOp�Conv2D_4/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�add/ReadVariableOp�add_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOp�batchnorm_1/ReadVariableOp�batchnorm_1/mul/ReadVariableOp�batchnorm_2/ReadVariableOp�batchnorm_2/mul/ReadVariableOp�batchnorm_3/ReadVariableOp�batchnorm_3/mul/ReadVariableOp�batchnorm_4/ReadVariableOp�batchnorm_4/mul/ReadVariableOp�batchnorm_5/ReadVariableOp�batchnorm_5/mul/ReadVariableOp�batchnorm_6/ReadVariableOp�batchnorm_6/mul/ReadVariableOp�batchnorm_7/ReadVariableOp�batchnorm_7/mul/ReadVariableOp�batchnorm_8/ReadVariableOp�batchnorm_8/mul/ReadVariableOp�depthwise/ReadVariableOp�depthwise_1/ReadVariableOp�depthwise_2/ReadVariableOp�depthwise_3/ReadVariableOp�#dropout/GreaterEqual/ReadVariableOp�dropout/Sub/ReadVariableOp�%dropout_1/GreaterEqual/ReadVariableOp�dropout_1/Sub/ReadVariableOp�%dropout_2/GreaterEqual/ReadVariableOp�dropout_2/Sub/ReadVariableOp�%dropout_3/GreaterEqual/ReadVariableOp�dropout_3/Sub/ReadVariableOp�%dropout_4/GreaterEqual/ReadVariableOp�dropout_4/Sub/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:����������� *
paddingSAME*
strides
Y
ReluReluConv2D:output:0*
T0*1
_output_shapes
:����������� s
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          �
moments/meanMeanRelu:activations:0'moments/mean/reduction_indices:output:0*
T0*&
_output_shapes
: *
	keep_dims(l
moments/StopGradientStopGradientmoments/mean:output:0*
T0*&
_output_shapes
: �
moments/SquaredDifferenceSquaredDifferenceRelu:activations:0moments/StopGradient:output:0*
T0*1
_output_shapes
:����������� w
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*&
_output_shapes
: *
	keep_dims(o
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 u
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: z
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: y
batchnorm/mul_1MulRelu:activations:0batchnorm/mul:z:0*
T0*1
_output_shapes
:����������� h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: r
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: |
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*1
_output_shapes
:����������� R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?v
dropout/Sub/ReadVariableOpReadVariableOp#dropout_sub_readvariableop_resource*
_output_shapes
: *
dtype0o
dropout/SubSubdropout/Const:output:0"dropout/Sub/ReadVariableOp:value:0*
T0*
_output_shapes
: |
dropout/RealDivRealDivbatchnorm/add_1:z:0dropout/Sub:z:0*
T0*1
_output_shapes
:����������� P
dropout/ShapeShapebatchnorm/add_1:z:0*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*1
_output_shapes
:����������� *
dtype0
#dropout/GreaterEqual/ReadVariableOpReadVariableOp#dropout_sub_readvariableop_resource*
_output_shapes
: *
dtype0�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0+dropout/GreaterEqual/ReadVariableOp:value:0*
T0*1
_output_shapes
:����������� T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/RealDiv:z:0dropout/Const_1:output:0*
T0*1
_output_shapes
:����������� �
depthwise/ReadVariableOpReadVariableOp!depthwise_readvariableop_resource*&
_output_shapes
: *
dtype0h
depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             h
depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
	depthwiseDepthwiseConv2dNativedropout/SelectV2:output:0 depthwise/ReadVariableOp:value:0*
T0*1
_output_shapes
:����������� *
paddingSAME*
strides
^
Relu_1Reludepthwise:output:0*
T0*1
_output_shapes
:����������� u
 moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          �
moments_1/meanMeanRelu_1:activations:0)moments_1/mean/reduction_indices:output:0*
T0*&
_output_shapes
: *
	keep_dims(p
moments_1/StopGradientStopGradientmoments_1/mean:output:0*
T0*&
_output_shapes
: �
moments_1/SquaredDifferenceSquaredDifferenceRelu_1:activations:0moments_1/StopGradient:output:0*
T0*1
_output_shapes
:����������� y
$moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          �
moments_1/varianceMeanmoments_1/SquaredDifference:z:0-moments_1/variance/reduction_indices:output:0*
T0*&
_output_shapes
: *
	keep_dims(s
moments_1/SqueezeSqueezemoments_1/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 y
moments_1/Squeeze_1Squeezemoments_1/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 V
batchnorm_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm_1/addAddV2moments_1/Squeeze_1:output:0batchnorm_1/add/y:output:0*
T0*
_output_shapes
: T
batchnorm_1/RsqrtRsqrtbatchnorm_1/add:z:0*
T0*
_output_shapes
: ~
batchnorm_1/mul/ReadVariableOpReadVariableOp'batchnorm_1_mul_readvariableop_resource*
_output_shapes
: *
dtype0z
batchnorm_1/mulMulbatchnorm_1/Rsqrt:y:0&batchnorm_1/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 
batchnorm_1/mul_1MulRelu_1:activations:0batchnorm_1/mul:z:0*
T0*1
_output_shapes
:����������� n
batchnorm_1/mul_2Mulmoments_1/Squeeze:output:0batchnorm_1/mul:z:0*
T0*
_output_shapes
: v
batchnorm_1/ReadVariableOpReadVariableOp#batchnorm_1_readvariableop_resource*
_output_shapes
: *
dtype0v
batchnorm_1/subSub"batchnorm_1/ReadVariableOp:value:0batchnorm_1/mul_2:z:0*
T0*
_output_shapes
: �
batchnorm_1/add_1AddV2batchnorm_1/mul_1:z:0batchnorm_1/sub:z:0*
T0*1
_output_shapes
:����������� �
Conv2D_1/ReadVariableOpReadVariableOp conv2d_1_readvariableop_resource*&
_output_shapes
: @*
dtype0�
Conv2D_1Conv2Dbatchnorm_1/add_1:z:0Conv2D_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@*
paddingSAME*
strides
]
Relu_2ReluConv2D_1:output:0*
T0*1
_output_shapes
:�����������@u
 moments_2/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          �
moments_2/meanMeanRelu_2:activations:0)moments_2/mean/reduction_indices:output:0*
T0*&
_output_shapes
:@*
	keep_dims(p
moments_2/StopGradientStopGradientmoments_2/mean:output:0*
T0*&
_output_shapes
:@�
moments_2/SquaredDifferenceSquaredDifferenceRelu_2:activations:0moments_2/StopGradient:output:0*
T0*1
_output_shapes
:�����������@y
$moments_2/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          �
moments_2/varianceMeanmoments_2/SquaredDifference:z:0-moments_2/variance/reduction_indices:output:0*
T0*&
_output_shapes
:@*
	keep_dims(s
moments_2/SqueezeSqueezemoments_2/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 y
moments_2/Squeeze_1Squeezemoments_2/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 V
batchnorm_2/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm_2/addAddV2moments_2/Squeeze_1:output:0batchnorm_2/add/y:output:0*
T0*
_output_shapes
:@T
batchnorm_2/RsqrtRsqrtbatchnorm_2/add:z:0*
T0*
_output_shapes
:@~
batchnorm_2/mul/ReadVariableOpReadVariableOp'batchnorm_2_mul_readvariableop_resource*
_output_shapes
: *
dtype0z
batchnorm_2/mulMulbatchnorm_2/Rsqrt:y:0&batchnorm_2/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@
batchnorm_2/mul_1MulRelu_2:activations:0batchnorm_2/mul:z:0*
T0*1
_output_shapes
:�����������@n
batchnorm_2/mul_2Mulmoments_2/Squeeze:output:0batchnorm_2/mul:z:0*
T0*
_output_shapes
:@v
batchnorm_2/ReadVariableOpReadVariableOp#batchnorm_2_readvariableop_resource*
_output_shapes
: *
dtype0v
batchnorm_2/subSub"batchnorm_2/ReadVariableOp:value:0batchnorm_2/mul_2:z:0*
T0*
_output_shapes
:@�
batchnorm_2/add_1AddV2batchnorm_2/mul_1:z:0batchnorm_2/sub:z:0*
T0*1
_output_shapes
:�����������@T
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?z
dropout_1/Sub/ReadVariableOpReadVariableOp%dropout_1_sub_readvariableop_resource*
_output_shapes
: *
dtype0u
dropout_1/SubSubdropout_1/Const:output:0$dropout_1/Sub/ReadVariableOp:value:0*
T0*
_output_shapes
: �
dropout_1/RealDivRealDivbatchnorm_2/add_1:z:0dropout_1/Sub:z:0*
T0*1
_output_shapes
:�����������@T
dropout_1/ShapeShapebatchnorm_2/add_1:z:0*
T0*
_output_shapes
:�
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*1
_output_shapes
:�����������@*
dtype0�
%dropout_1/GreaterEqual/ReadVariableOpReadVariableOp%dropout_1_sub_readvariableop_resource*
_output_shapes
: *
dtype0�
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0-dropout_1/GreaterEqual/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@V
dropout_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout_1/SelectV2SelectV2dropout_1/GreaterEqual:z:0dropout_1/RealDiv:z:0dropout_1/Const_1:output:0*
T0*1
_output_shapes
:�����������@�
depthwise_1/ReadVariableOpReadVariableOp#depthwise_1_readvariableop_resource*&
_output_shapes
:@*
dtype0j
depthwise_1/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      @      j
depthwise_1/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
depthwise_1DepthwiseConv2dNativedropout_1/SelectV2:output:0"depthwise_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������KK@*
paddingSAME*
strides
^
Relu_3Reludepthwise_1:output:0*
T0*/
_output_shapes
:���������KK@u
 moments_3/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          �
moments_3/meanMeanRelu_3:activations:0)moments_3/mean/reduction_indices:output:0*
T0*&
_output_shapes
:@*
	keep_dims(p
moments_3/StopGradientStopGradientmoments_3/mean:output:0*
T0*&
_output_shapes
:@�
moments_3/SquaredDifferenceSquaredDifferenceRelu_3:activations:0moments_3/StopGradient:output:0*
T0*/
_output_shapes
:���������KK@y
$moments_3/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          �
moments_3/varianceMeanmoments_3/SquaredDifference:z:0-moments_3/variance/reduction_indices:output:0*
T0*&
_output_shapes
:@*
	keep_dims(s
moments_3/SqueezeSqueezemoments_3/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 y
moments_3/Squeeze_1Squeezemoments_3/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 V
batchnorm_3/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm_3/addAddV2moments_3/Squeeze_1:output:0batchnorm_3/add/y:output:0*
T0*
_output_shapes
:@T
batchnorm_3/RsqrtRsqrtbatchnorm_3/add:z:0*
T0*
_output_shapes
:@~
batchnorm_3/mul/ReadVariableOpReadVariableOp'batchnorm_3_mul_readvariableop_resource*
_output_shapes
: *
dtype0z
batchnorm_3/mulMulbatchnorm_3/Rsqrt:y:0&batchnorm_3/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@}
batchnorm_3/mul_1MulRelu_3:activations:0batchnorm_3/mul:z:0*
T0*/
_output_shapes
:���������KK@n
batchnorm_3/mul_2Mulmoments_3/Squeeze:output:0batchnorm_3/mul:z:0*
T0*
_output_shapes
:@v
batchnorm_3/ReadVariableOpReadVariableOp#batchnorm_3_readvariableop_resource*
_output_shapes
: *
dtype0v
batchnorm_3/subSub"batchnorm_3/ReadVariableOp:value:0batchnorm_3/mul_2:z:0*
T0*
_output_shapes
:@�
batchnorm_3/add_1AddV2batchnorm_3/mul_1:z:0batchnorm_3/sub:z:0*
T0*/
_output_shapes
:���������KK@�
Conv2D_2/ReadVariableOpReadVariableOp conv2d_2_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
Conv2D_2Conv2Dbatchnorm_3/add_1:z:0Conv2D_2/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������KK�*
paddingSAME*
strides
\
Relu_4ReluConv2D_2:output:0*
T0*0
_output_shapes
:���������KK�u
 moments_4/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          �
moments_4/meanMeanRelu_4:activations:0)moments_4/mean/reduction_indices:output:0*
T0*'
_output_shapes
:�*
	keep_dims(q
moments_4/StopGradientStopGradientmoments_4/mean:output:0*
T0*'
_output_shapes
:��
moments_4/SquaredDifferenceSquaredDifferenceRelu_4:activations:0moments_4/StopGradient:output:0*
T0*0
_output_shapes
:���������KK�y
$moments_4/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          �
moments_4/varianceMeanmoments_4/SquaredDifference:z:0-moments_4/variance/reduction_indices:output:0*
T0*'
_output_shapes
:�*
	keep_dims(t
moments_4/SqueezeSqueezemoments_4/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 z
moments_4/Squeeze_1Squeezemoments_4/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 V
batchnorm_4/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:x
batchnorm_4/addAddV2moments_4/Squeeze_1:output:0batchnorm_4/add/y:output:0*
T0*
_output_shapes	
:�U
batchnorm_4/RsqrtRsqrtbatchnorm_4/add:z:0*
T0*
_output_shapes	
:�~
batchnorm_4/mul/ReadVariableOpReadVariableOp'batchnorm_4_mul_readvariableop_resource*
_output_shapes
: *
dtype0{
batchnorm_4/mulMulbatchnorm_4/Rsqrt:y:0&batchnorm_4/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�~
batchnorm_4/mul_1MulRelu_4:activations:0batchnorm_4/mul:z:0*
T0*0
_output_shapes
:���������KK�o
batchnorm_4/mul_2Mulmoments_4/Squeeze:output:0batchnorm_4/mul:z:0*
T0*
_output_shapes	
:�v
batchnorm_4/ReadVariableOpReadVariableOp#batchnorm_4_readvariableop_resource*
_output_shapes
: *
dtype0w
batchnorm_4/subSub"batchnorm_4/ReadVariableOp:value:0batchnorm_4/mul_2:z:0*
T0*
_output_shapes	
:��
batchnorm_4/add_1AddV2batchnorm_4/mul_1:z:0batchnorm_4/sub:z:0*
T0*0
_output_shapes
:���������KK�T
dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?z
dropout_2/Sub/ReadVariableOpReadVariableOp%dropout_2_sub_readvariableop_resource*
_output_shapes
: *
dtype0u
dropout_2/SubSubdropout_2/Const:output:0$dropout_2/Sub/ReadVariableOp:value:0*
T0*
_output_shapes
: �
dropout_2/RealDivRealDivbatchnorm_4/add_1:z:0dropout_2/Sub:z:0*
T0*0
_output_shapes
:���������KK�T
dropout_2/ShapeShapebatchnorm_4/add_1:z:0*
T0*
_output_shapes
:�
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*0
_output_shapes
:���������KK�*
dtype0�
%dropout_2/GreaterEqual/ReadVariableOpReadVariableOp%dropout_2_sub_readvariableop_resource*
_output_shapes
: *
dtype0�
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0-dropout_2/GreaterEqual/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������KK�V
dropout_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout_2/SelectV2SelectV2dropout_2/GreaterEqual:z:0dropout_2/RealDiv:z:0dropout_2/Const_1:output:0*
T0*0
_output_shapes
:���������KK��
depthwise_2/ReadVariableOpReadVariableOp#depthwise_2_readvariableop_resource*'
_output_shapes
:�*
dtype0j
depthwise_2/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      �      j
depthwise_2/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
depthwise_2DepthwiseConv2dNativedropout_2/SelectV2:output:0"depthwise_2/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������&&�*
paddingSAME*
strides
_
Relu_5Reludepthwise_2:output:0*
T0*0
_output_shapes
:���������&&�u
 moments_5/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          �
moments_5/meanMeanRelu_5:activations:0)moments_5/mean/reduction_indices:output:0*
T0*'
_output_shapes
:�*
	keep_dims(q
moments_5/StopGradientStopGradientmoments_5/mean:output:0*
T0*'
_output_shapes
:��
moments_5/SquaredDifferenceSquaredDifferenceRelu_5:activations:0moments_5/StopGradient:output:0*
T0*0
_output_shapes
:���������&&�y
$moments_5/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          �
moments_5/varianceMeanmoments_5/SquaredDifference:z:0-moments_5/variance/reduction_indices:output:0*
T0*'
_output_shapes
:�*
	keep_dims(t
moments_5/SqueezeSqueezemoments_5/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 z
moments_5/Squeeze_1Squeezemoments_5/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 V
batchnorm_5/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:x
batchnorm_5/addAddV2moments_5/Squeeze_1:output:0batchnorm_5/add/y:output:0*
T0*
_output_shapes	
:�U
batchnorm_5/RsqrtRsqrtbatchnorm_5/add:z:0*
T0*
_output_shapes	
:�~
batchnorm_5/mul/ReadVariableOpReadVariableOp'batchnorm_5_mul_readvariableop_resource*
_output_shapes
: *
dtype0{
batchnorm_5/mulMulbatchnorm_5/Rsqrt:y:0&batchnorm_5/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�~
batchnorm_5/mul_1MulRelu_5:activations:0batchnorm_5/mul:z:0*
T0*0
_output_shapes
:���������&&�o
batchnorm_5/mul_2Mulmoments_5/Squeeze:output:0batchnorm_5/mul:z:0*
T0*
_output_shapes	
:�v
batchnorm_5/ReadVariableOpReadVariableOp#batchnorm_5_readvariableop_resource*
_output_shapes
: *
dtype0w
batchnorm_5/subSub"batchnorm_5/ReadVariableOp:value:0batchnorm_5/mul_2:z:0*
T0*
_output_shapes	
:��
batchnorm_5/add_1AddV2batchnorm_5/mul_1:z:0batchnorm_5/sub:z:0*
T0*0
_output_shapes
:���������&&��
Conv2D_3/ReadVariableOpReadVariableOp conv2d_3_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2D_3Conv2Dbatchnorm_5/add_1:z:0Conv2D_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������&&�*
paddingSAME*
strides
\
Relu_6ReluConv2D_3:output:0*
T0*0
_output_shapes
:���������&&�u
 moments_6/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          �
moments_6/meanMeanRelu_6:activations:0)moments_6/mean/reduction_indices:output:0*
T0*'
_output_shapes
:�*
	keep_dims(q
moments_6/StopGradientStopGradientmoments_6/mean:output:0*
T0*'
_output_shapes
:��
moments_6/SquaredDifferenceSquaredDifferenceRelu_6:activations:0moments_6/StopGradient:output:0*
T0*0
_output_shapes
:���������&&�y
$moments_6/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          �
moments_6/varianceMeanmoments_6/SquaredDifference:z:0-moments_6/variance/reduction_indices:output:0*
T0*'
_output_shapes
:�*
	keep_dims(t
moments_6/SqueezeSqueezemoments_6/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 z
moments_6/Squeeze_1Squeezemoments_6/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 V
batchnorm_6/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:x
batchnorm_6/addAddV2moments_6/Squeeze_1:output:0batchnorm_6/add/y:output:0*
T0*
_output_shapes	
:�U
batchnorm_6/RsqrtRsqrtbatchnorm_6/add:z:0*
T0*
_output_shapes	
:�~
batchnorm_6/mul/ReadVariableOpReadVariableOp'batchnorm_6_mul_readvariableop_resource*
_output_shapes
: *
dtype0{
batchnorm_6/mulMulbatchnorm_6/Rsqrt:y:0&batchnorm_6/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�~
batchnorm_6/mul_1MulRelu_6:activations:0batchnorm_6/mul:z:0*
T0*0
_output_shapes
:���������&&�o
batchnorm_6/mul_2Mulmoments_6/Squeeze:output:0batchnorm_6/mul:z:0*
T0*
_output_shapes	
:�v
batchnorm_6/ReadVariableOpReadVariableOp#batchnorm_6_readvariableop_resource*
_output_shapes
: *
dtype0w
batchnorm_6/subSub"batchnorm_6/ReadVariableOp:value:0batchnorm_6/mul_2:z:0*
T0*
_output_shapes	
:��
batchnorm_6/add_1AddV2batchnorm_6/mul_1:z:0batchnorm_6/sub:z:0*
T0*0
_output_shapes
:���������&&�T
dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?z
dropout_3/Sub/ReadVariableOpReadVariableOp%dropout_3_sub_readvariableop_resource*
_output_shapes
: *
dtype0u
dropout_3/SubSubdropout_3/Const:output:0$dropout_3/Sub/ReadVariableOp:value:0*
T0*
_output_shapes
: �
dropout_3/RealDivRealDivbatchnorm_6/add_1:z:0dropout_3/Sub:z:0*
T0*0
_output_shapes
:���������&&�T
dropout_3/ShapeShapebatchnorm_6/add_1:z:0*
T0*
_output_shapes
:�
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*0
_output_shapes
:���������&&�*
dtype0�
%dropout_3/GreaterEqual/ReadVariableOpReadVariableOp%dropout_3_sub_readvariableop_resource*
_output_shapes
: *
dtype0�
dropout_3/GreaterEqualGreaterEqual/dropout_3/random_uniform/RandomUniform:output:0-dropout_3/GreaterEqual/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������&&�V
dropout_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout_3/SelectV2SelectV2dropout_3/GreaterEqual:z:0dropout_3/RealDiv:z:0dropout_3/Const_1:output:0*
T0*0
_output_shapes
:���������&&��
depthwise_3/ReadVariableOpReadVariableOp#depthwise_3_readvariableop_resource*'
_output_shapes
:�*
dtype0j
depthwise_3/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            j
depthwise_3/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
depthwise_3DepthwiseConv2dNativedropout_3/SelectV2:output:0"depthwise_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
_
Relu_7Reludepthwise_3:output:0*
T0*0
_output_shapes
:����������u
 moments_7/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          �
moments_7/meanMeanRelu_7:activations:0)moments_7/mean/reduction_indices:output:0*
T0*'
_output_shapes
:�*
	keep_dims(q
moments_7/StopGradientStopGradientmoments_7/mean:output:0*
T0*'
_output_shapes
:��
moments_7/SquaredDifferenceSquaredDifferenceRelu_7:activations:0moments_7/StopGradient:output:0*
T0*0
_output_shapes
:����������y
$moments_7/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          �
moments_7/varianceMeanmoments_7/SquaredDifference:z:0-moments_7/variance/reduction_indices:output:0*
T0*'
_output_shapes
:�*
	keep_dims(t
moments_7/SqueezeSqueezemoments_7/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 z
moments_7/Squeeze_1Squeezemoments_7/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 V
batchnorm_7/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:x
batchnorm_7/addAddV2moments_7/Squeeze_1:output:0batchnorm_7/add/y:output:0*
T0*
_output_shapes	
:�U
batchnorm_7/RsqrtRsqrtbatchnorm_7/add:z:0*
T0*
_output_shapes	
:�~
batchnorm_7/mul/ReadVariableOpReadVariableOp'batchnorm_7_mul_readvariableop_resource*
_output_shapes
: *
dtype0{
batchnorm_7/mulMulbatchnorm_7/Rsqrt:y:0&batchnorm_7/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�~
batchnorm_7/mul_1MulRelu_7:activations:0batchnorm_7/mul:z:0*
T0*0
_output_shapes
:����������o
batchnorm_7/mul_2Mulmoments_7/Squeeze:output:0batchnorm_7/mul:z:0*
T0*
_output_shapes	
:�v
batchnorm_7/ReadVariableOpReadVariableOp#batchnorm_7_readvariableop_resource*
_output_shapes
: *
dtype0w
batchnorm_7/subSub"batchnorm_7/ReadVariableOp:value:0batchnorm_7/mul_2:z:0*
T0*
_output_shapes	
:��
batchnorm_7/add_1AddV2batchnorm_7/mul_1:z:0batchnorm_7/sub:z:0*
T0*0
_output_shapes
:�����������
Conv2D_4/ReadVariableOpReadVariableOp conv2d_4_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2D_4Conv2Dbatchnorm_7/add_1:z:0Conv2D_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
\
Relu_8ReluConv2D_4:output:0*
T0*0
_output_shapes
:����������u
 moments_8/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          �
moments_8/meanMeanRelu_8:activations:0)moments_8/mean/reduction_indices:output:0*
T0*'
_output_shapes
:�*
	keep_dims(q
moments_8/StopGradientStopGradientmoments_8/mean:output:0*
T0*'
_output_shapes
:��
moments_8/SquaredDifferenceSquaredDifferenceRelu_8:activations:0moments_8/StopGradient:output:0*
T0*0
_output_shapes
:����������y
$moments_8/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          �
moments_8/varianceMeanmoments_8/SquaredDifference:z:0-moments_8/variance/reduction_indices:output:0*
T0*'
_output_shapes
:�*
	keep_dims(t
moments_8/SqueezeSqueezemoments_8/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 z
moments_8/Squeeze_1Squeezemoments_8/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 V
batchnorm_8/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:x
batchnorm_8/addAddV2moments_8/Squeeze_1:output:0batchnorm_8/add/y:output:0*
T0*
_output_shapes	
:�U
batchnorm_8/RsqrtRsqrtbatchnorm_8/add:z:0*
T0*
_output_shapes	
:�~
batchnorm_8/mul/ReadVariableOpReadVariableOp'batchnorm_8_mul_readvariableop_resource*
_output_shapes
: *
dtype0{
batchnorm_8/mulMulbatchnorm_8/Rsqrt:y:0&batchnorm_8/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�~
batchnorm_8/mul_1MulRelu_8:activations:0batchnorm_8/mul:z:0*
T0*0
_output_shapes
:����������o
batchnorm_8/mul_2Mulmoments_8/Squeeze:output:0batchnorm_8/mul:z:0*
T0*
_output_shapes	
:�v
batchnorm_8/ReadVariableOpReadVariableOp#batchnorm_8_readvariableop_resource*
_output_shapes
: *
dtype0w
batchnorm_8/subSub"batchnorm_8/ReadVariableOp:value:0batchnorm_8/mul_2:z:0*
T0*
_output_shapes	
:��
batchnorm_8/add_1AddV2batchnorm_8/mul_1:z:0batchnorm_8/sub:z:0*
T0*0
_output_shapes
:����������T
dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?z
dropout_4/Sub/ReadVariableOpReadVariableOp%dropout_4_sub_readvariableop_resource*
_output_shapes
: *
dtype0u
dropout_4/SubSubdropout_4/Const:output:0$dropout_4/Sub/ReadVariableOp:value:0*
T0*
_output_shapes
: �
dropout_4/RealDivRealDivbatchnorm_8/add_1:z:0dropout_4/Sub:z:0*
T0*0
_output_shapes
:����������T
dropout_4/ShapeShapebatchnorm_8/add_1:z:0*
T0*
_output_shapes
:�
&dropout_4/random_uniform/RandomUniformRandomUniformdropout_4/Shape:output:0*
T0*0
_output_shapes
:����������*
dtype0�
%dropout_4/GreaterEqual/ReadVariableOpReadVariableOp%dropout_4_sub_readvariableop_resource*
_output_shapes
: *
dtype0�
dropout_4/GreaterEqualGreaterEqual/dropout_4/random_uniform/RandomUniform:output:0-dropout_4/GreaterEqual/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������V
dropout_4/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout_4/SelectV2SelectV2dropout_4/GreaterEqual:z:0dropout_4/RealDiv:z:0dropout_4/Const_1:output:0*
T0*0
_output_shapes
:����������g
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      }
MeanMeandropout_4/SelectV2:output:0Mean/reduction_indices:output:0*
T0*(
_output_shapes
:����������v
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0q
MatMulMatMulMean:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������k
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes	
:�*
dtype0m
addAddV2MatMul:product:0add/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������J
Relu_9Reluadd:z:0*
T0*(
_output_shapes
:����������y
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	�e*
dtype0{
MatMul_1MatMulRelu_9:activations:0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������en
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
:e*
dtype0r
add_1AddV2MatMul_1:product:0add_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������eO
SoftmaxSoftmax	add_1:z:0*
T0*'
_output_shapes
:���������e`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������e�

NoOpNoOp^Conv2D/ReadVariableOp^Conv2D_1/ReadVariableOp^Conv2D_2/ReadVariableOp^Conv2D_3/ReadVariableOp^Conv2D_4/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^add/ReadVariableOp^add_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp^batchnorm_1/ReadVariableOp^batchnorm_1/mul/ReadVariableOp^batchnorm_2/ReadVariableOp^batchnorm_2/mul/ReadVariableOp^batchnorm_3/ReadVariableOp^batchnorm_3/mul/ReadVariableOp^batchnorm_4/ReadVariableOp^batchnorm_4/mul/ReadVariableOp^batchnorm_5/ReadVariableOp^batchnorm_5/mul/ReadVariableOp^batchnorm_6/ReadVariableOp^batchnorm_6/mul/ReadVariableOp^batchnorm_7/ReadVariableOp^batchnorm_7/mul/ReadVariableOp^batchnorm_8/ReadVariableOp^batchnorm_8/mul/ReadVariableOp^depthwise/ReadVariableOp^depthwise_1/ReadVariableOp^depthwise_2/ReadVariableOp^depthwise_3/ReadVariableOp$^dropout/GreaterEqual/ReadVariableOp^dropout/Sub/ReadVariableOp&^dropout_1/GreaterEqual/ReadVariableOp^dropout_1/Sub/ReadVariableOp&^dropout_2/GreaterEqual/ReadVariableOp^dropout_2/Sub/ReadVariableOp&^dropout_3/GreaterEqual/ReadVariableOp^dropout_3/Sub/ReadVariableOp&^dropout_4/GreaterEqual/ReadVariableOp^dropout_4/Sub/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp22
Conv2D_1/ReadVariableOpConv2D_1/ReadVariableOp22
Conv2D_2/ReadVariableOpConv2D_2/ReadVariableOp22
Conv2D_3/ReadVariableOpConv2D_3/ReadVariableOp22
Conv2D_4/ReadVariableOpConv2D_4/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp2,
add_1/ReadVariableOpadd_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp28
batchnorm_1/ReadVariableOpbatchnorm_1/ReadVariableOp2@
batchnorm_1/mul/ReadVariableOpbatchnorm_1/mul/ReadVariableOp28
batchnorm_2/ReadVariableOpbatchnorm_2/ReadVariableOp2@
batchnorm_2/mul/ReadVariableOpbatchnorm_2/mul/ReadVariableOp28
batchnorm_3/ReadVariableOpbatchnorm_3/ReadVariableOp2@
batchnorm_3/mul/ReadVariableOpbatchnorm_3/mul/ReadVariableOp28
batchnorm_4/ReadVariableOpbatchnorm_4/ReadVariableOp2@
batchnorm_4/mul/ReadVariableOpbatchnorm_4/mul/ReadVariableOp28
batchnorm_5/ReadVariableOpbatchnorm_5/ReadVariableOp2@
batchnorm_5/mul/ReadVariableOpbatchnorm_5/mul/ReadVariableOp28
batchnorm_6/ReadVariableOpbatchnorm_6/ReadVariableOp2@
batchnorm_6/mul/ReadVariableOpbatchnorm_6/mul/ReadVariableOp28
batchnorm_7/ReadVariableOpbatchnorm_7/ReadVariableOp2@
batchnorm_7/mul/ReadVariableOpbatchnorm_7/mul/ReadVariableOp28
batchnorm_8/ReadVariableOpbatchnorm_8/ReadVariableOp2@
batchnorm_8/mul/ReadVariableOpbatchnorm_8/mul/ReadVariableOp24
depthwise/ReadVariableOpdepthwise/ReadVariableOp28
depthwise_1/ReadVariableOpdepthwise_1/ReadVariableOp28
depthwise_2/ReadVariableOpdepthwise_2/ReadVariableOp28
depthwise_3/ReadVariableOpdepthwise_3/ReadVariableOp2J
#dropout/GreaterEqual/ReadVariableOp#dropout/GreaterEqual/ReadVariableOp28
dropout/Sub/ReadVariableOpdropout/Sub/ReadVariableOp2N
%dropout_1/GreaterEqual/ReadVariableOp%dropout_1/GreaterEqual/ReadVariableOp2<
dropout_1/Sub/ReadVariableOpdropout_1/Sub/ReadVariableOp2N
%dropout_2/GreaterEqual/ReadVariableOp%dropout_2/GreaterEqual/ReadVariableOp2<
dropout_2/Sub/ReadVariableOpdropout_2/Sub/ReadVariableOp2N
%dropout_3/GreaterEqual/ReadVariableOp%dropout_3/GreaterEqual/ReadVariableOp2<
dropout_3/Sub/ReadVariableOpdropout_3/Sub/ReadVariableOp2N
%dropout_4/GreaterEqual/ReadVariableOp%dropout_4/GreaterEqual/ReadVariableOp2<
dropout_4/Sub/ReadVariableOpdropout_4/Sub/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
&__inference_signature_wrapper_52924652

inputs!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: #
	unknown_3: 
	unknown_4: 
	unknown_5: #
	unknown_6: @
	unknown_7: 
	unknown_8: 
	unknown_9: $

unknown_10:@

unknown_11: 

unknown_12: %

unknown_13:@�

unknown_14: 

unknown_15: 

unknown_16: %

unknown_17:�

unknown_18: 

unknown_19: &

unknown_20:��

unknown_21: 

unknown_22: 

unknown_23: %

unknown_24:�

unknown_25: 

unknown_26: &

unknown_27:��

unknown_28: 

unknown_29: 

unknown_30: 

unknown_31:
��

unknown_32:	�

unknown_33:	�e

unknown_34:e
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������e*F
_read_only_resource_inputs(
&$	
 !"#$*-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference_serving_default_52924573o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������e`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs"�
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
C
inputs9
serving_default_inputs:0�����������<
output_00
StatefulPartitionedCall:0���������etensorflow/serving/predict:�
�
conv_block1
depthwise_conv_block1
depthwise_conv_block2
depthwise_conv_block3
depthwise_conv_block4
dense_weights1
dense_bias1
dense_weights2
	dense_bias2

serving_default

signatures"
_generic_user_object
E
conv_weights
bn
dropout"
_generic_user_object
i
depthwise_filter
pointwise_filter
bn1
bn2
dropout"
_generic_user_object
i
depthwise_filter
pointwise_filter
bn1
bn2
dropout"
_generic_user_object
i
depthwise_filter
pointwise_filter
bn1
bn2
dropout"
_generic_user_object
i
depthwise_filter
pointwise_filter
 bn1
!bn2
"dropout"
_generic_user_object
:
��2Variable
:�2Variable
:	�e2Variable
:e2Variable
�
#trace_02�
$__inference_serving_default_52924573�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *'�$
"������������z#trace_0
,
$serving_default"
signature_map
":  2Variable
5
	%scale

&offset"
_generic_user_object
: 2Variable
":  2Variable
":  @2Variable
5
	'scale

(offset"
_generic_user_object
5
	)scale

*offset"
_generic_user_object
: 2Variable
": @2Variable
#:!@�2Variable
5
	+scale

,offset"
_generic_user_object
5
	-scale

.offset"
_generic_user_object
: 2Variable
#:!�2Variable
$:"��2Variable
5
	/scale

0offset"
_generic_user_object
5
	1scale

2offset"
_generic_user_object
: 2Variable
#:!�2Variable
$:"��2Variable
5
	3scale

4offset"
_generic_user_object
5
	5scale

6offset"
_generic_user_object
: 2Variable
�B�
$__inference_serving_default_52924573inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *'�$
"������������
�B�
&__inference_signature_wrapper_52924652inputs"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
: 2Variable
: 2Variable
: 2Variable
: 2Variable
: 2Variable
: 2Variable
: 2Variable
: 2Variable
: 2Variable
: 2Variable
: 2Variable
: 2Variable
: 2Variable
: 2Variable
: 2Variable
: 2Variable
: 2Variable
: 2Variable�
$__inference_serving_default_52924573�$%&'()*+,-./0123456"	9�6
/�,
*�'
inputs�����������
� "!�
unknown���������e�
&__inference_signature_wrapper_52924652�$%&'()*+,-./0123456"	C�@
� 
9�6
4
inputs*�'
inputs�����������"3�0
.
output_0"�
output_0���������e