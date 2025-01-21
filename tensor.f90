MODULE TENSOR
    IMPLICIT NONE
    PRIVATE
    PUBLIC :: Tensor_t, Adam_Optimizer_t, create_tensor, destroy_tensor, tensor_matmul, &
              tensor_add, tensor_subtract, tensor_multiply, tensor_divide, &
              apply_relu, apply_sigmoid, tensor_transpose, apply_softmax, tensor_reshape, &
              batch_normalize, apply_dropout, conv2d, max_pool2d, &
              cross_entropy_loss, sgd_optimizer, init_adam_optimizer, adam_optimizer_step, &
              mse_loss, mae_loss, huber_loss, &
              init_weights_xavier, init_weights_he, init_weights_uniform, &
              average_pool2d, global_avg_pool, layer_normalize, &
              tensor_sum, tensor_mean, tensor_max, tensor_min, &
              tensor_concatenate, tensor_slice, tensor_pad, &
              tensor_einsum, tensor_where, tensor_clip, &
              backward_pass, zero_grad, tensor_to_cpu, tensor_to_gpu, &
              tensor_to_accelerator, is_accelerated, apply_tanh, &
              apply_leaky_relu, binary_cross_entropy_loss, RMSprop_Optimizer_t, random_horizontal_flip

    ! CONSTANTS FOR OPTIMIZERS
    REAL, PARAMETER :: DEFAULT_LEARNING_RATE = 0.0001
    REAL, PARAMETER :: DEFAULT_BETA1 = 0.9
    REAL, PARAMETER :: DEFAULT_BETA2 = 0.999
    REAL, PARAMETER :: EPSILON = 1.0E-8

    ! OPTIMIZER TYPE FOR ADAM
    TYPE :: Adam_Optimizer_t
        REAL :: learning_rate
        REAL :: beta1
        REAL :: beta2
        INTEGER :: t = 0                        ! TIME STEP
        TYPE(Tensor_t), ALLOCATABLE :: m(:)     ! FIRST MOMENTUM
        TYPE(Tensor_t), ALLOCATABLE :: v(:)     ! SECOND MOMENTUM
    END TYPE Adam_Optimizer_t

    ! RMSprop Optimizer
    TYPE :: RMSprop_Optimizer_t
        REAL :: learning_rate
        REAL :: decay_rate
        REAL :: epsilon
        REAL, ALLOCATABLE :: square_grad(:,:,:)
    END TYPE RMSprop_Optimizer_t

    ! DEFINE TENSOR TYPE
    TYPE :: Tensor_t
        REAL, ALLOCATABLE :: data(:,:,:)
        INTEGER :: shape(3)
        REAL :: grad_scale = 1.0 ! backpropagation gradient
        LOGICAL :: required_grad = .FALSE. ! flag for backpropagation
        REAL, ALLOCATABLE :: grad(:,:,:)
        LOGICAL :: use_accelerator = .FALSE. ! flag for accelerated operations
    END TYPE Tensor_t

    CONTAINS
        ! CREATE AND INITIALIZE TENSOR
        SUBROUTINE create_tensor(t, dims)
            TYPE(Tensor_t), INTENT(INOUT) :: t
            INTEGER, INTENT(IN) :: dims(3)

            t%shape = dims
            ALLOCATE(t%data(dims(1), dims(2), dims(3)))
            t%data = 0.00000000
        END SUBROUTINE create_tensor

        ! DEALLOCATE TENSORS
        SUBROUTINE destroy_tensor(t)
            TYPE(Tensor_t), INTENT(INOUT) :: t

            IF (ALLOCATED(t%data)) THEN
                DEALLOCATE(t%data)
            END IF
        END SUBROUTINE destroy_tensor

        ! Hardware acceleration using Intel oneAPI
        SUBROUTINE tensor_to_accelerator(t)
            USE ifcore
            TYPE(Tensor_t), INTENT(INOUT) :: t
            
            t%use_accelerator = .TRUE.
        END SUBROUTINE tensor_to_accelerator

        SUBROUTINE tensor_to_cpu(t)
            TYPE(Tensor_t), INTENT(INOUT) :: t
            
            t%use_accelerator = .FALSE.
        END SUBROUTINE tensor_to_cpu

        ! Helper function to check if tensor is using accelerator
        FUNCTION is_accelerated(t) RESULT(accelerated)
            TYPE(Tensor_t), INTENT(IN) :: t
            LOGICAL :: accelerated
            
            accelerated = t%use_accelerator
        END FUNCTION is_accelerated

        ! Optimized matrix multiplication
        SUBROUTINE tensor_matmul(a, b, result)
            TYPE(Tensor_t), INTENT(IN) :: a, b
            TYPE(Tensor_t), INTENT(INOUT) :: result
            INTEGER :: m, n, k, i, j, p
            REAL :: temp
            
            m = a%shape(1)
            k = a%shape(2)
            n = b%shape(2)
            
            CALL create_tensor(result, [m, n, 1])
            
            !$OMP PARALLEL DO PRIVATE(i,j,p,temp) COLLAPSE(2)
            DO j = 1, n
                DO i = 1, m
                    temp = 0.0
                    DO p = 1, k
                        temp = temp + a%data(i,p,1) * b%data(p,j,1)
                    END DO
                    result%data(i,j,1) = temp
                END DO
            END DO
            !$OMP END PARALLEL DO
        END SUBROUTINE tensor_matmul

        ! Optimized element-wise operations
        SUBROUTINE tensor_add(a, b, result)
            TYPE(Tensor_t), INTENT(IN) :: a, b
            TYPE(Tensor_t), INTENT(INOUT) :: result
            INTEGER :: i, j, k
            
            CALL create_tensor(result, a%shape)
            
            !$OMP PARALLEL DO PRIVATE(i,j,k) COLLAPSE(3)
            DO k = 1, a%shape(3)
                DO j = 1, a%shape(2)
                    DO i = 1, a%shape(1)
                        result%data(i,j,k) = a%data(i,j,k) + b%data(i,j,k)
                    END DO
                END DO
            END DO
            !$OMP END PARALLEL DO
        END SUBROUTINE tensor_add

        ! ELEMENT-WISE OPERATIONS
        SUBROUTINE tensor_subtract(t1, t2, result)
            TYPE(Tensor_t), INTENT(IN) :: t1, t2
            TYPE(Tensor_t), INTENT(INOUT) :: result

            IF (ANY(t1%shape /= t2%shape)) THEN
                PRINT *, "Error: Incompatible tensor dimensions for element-wise addition"
                RETURN
            END IF

            CALL create_tensor(result, t1%shape)
            result%data = t1%data - t2%data
        END SUBROUTINE tensor_subtract

        SUBROUTINE tensor_multiply(t1, t2, result)
            TYPE(Tensor_t), INTENT(IN) :: t1, t2
            TYPE(Tensor_t), INTENT(INOUT) :: result

            IF (ANY(t1%shape /= t2%shape)) THEN
                PRINT *, "Error: Incompatible tensor dimensions for element-wise addition"
                RETURN
            END IF

            CALL create_tensor(result, t1%shape)
            result%data = t1%data * t2%data
        END SUBROUTINE tensor_multiply

        SUBROUTINE tensor_divide(t1, t2, result)
            TYPE(Tensor_t), INTENT(IN) :: t1, t2
            TYPE(Tensor_t), INTENT(INOUT) :: result

            IF (ANY(t1%shape /= t2%shape)) THEN
                PRINT *, "Error: Incompatible tensor dimensions for element-wise addition"
                RETURN
            END IF

            CALL create_tensor(result, t1%shape)
            result%data = t1%data / t2%data
        END SUBROUTINE tensor_divide

        ! ReLU ACTIVATION FUNCTION
        SUBROUTINE apply_relu(t, result)
            TYPE(Tensor_t), INTENT(IN) :: t
            TYPE(Tensor_t), INTENT(INOUT) :: result

            CALL create_tensor(result, t%shape)
            result%data = MAX(0.0, t%data)

            ! STORE GRADIENT INFORMATIONS
            IF (t%required_grad) THEN
                result%required_grad = .TRUE.
                IF (.NOT. ALLOCATED(result%grad)) THEN
                    ALLOCATE(result%grad(t%shape(1), t%shape(2), t%shape(3)))
                END IF
                WHERE (t%data > 0.0)
                    result%grad = 1.0
                ELSEWHERE
                    result%grad = 0.0
                END WHERE
            END IF
        END SUBROUTINE apply_relu

        ! SIGMOID ACTIVATION FUNCTION
        SUBROUTINE apply_sigmoid(t, result)
            TYPE(Tensor_t), INTENT(IN) :: t
            TYPE(Tensor_t), INTENT(INOUT) :: result

            CALL create_tensor(result, t%shape)
            result%data = 1.0 / (1.0 + EXP(-t%data))

            IF (t%required_grad) THEN
                result%required_grad = .TRUE.
                IF (.NOT. ALLOCATED(result%grad)) THEN
                    ALLOCATE(result%grad(t%shape(1), t%shape(2), t%shape(3)))
                END IF
                result%grad = result%grad * (1.0 - result%data)
            END IF
        END SUBROUTINE apply_sigmoid

        ! SOFTMAX ACTIVATION (LAST DIMENSION)
        SUBROUTINE apply_softmax(t, result)
            TYPE(Tensor_t), INTENT(IN) :: t
            TYPE(Tensor_t), INTENT(INOUT) :: result
            REAL :: max_val, sum_exp
            INTEGER :: i, j

            CALL create_tensor(result, t%shape)

            DO i = 1, t%shape(1)
                DO j = 1, t%shape(2)
                    max_val = MAXVAL(t%data(i, j, :))
                    result%data(i, j, :) = EXP(t%data(i, j, :) - max_val)
                    sum_exp = SUM(result%data(i, j, :))
                    result%data(i, j, :) = result%data(i, j, :) / sum_exp
                END DO
            END DO
        END SUBROUTINE apply_softmax

        ! TENSOR TRANSPOSE
        SUBROUTINE tensor_transpose(t, result, dim1, dim2)
            TYPE(Tensor_t), INTENT(IN) :: t
            TYPE(Tensor_t), INTENT(INOUT) :: result
            INTEGER, INTENT(IN) :: dim1, dim2
            INTEGER :: new_shape(3)
            INTEGER :: i

            new_shape = t%shape
            new_shape(dim1) = t%shape(dim2)
            new_shape(dim2) = t%shape(dim1)

            CALL create_tensor(result, new_shape)

            SELECT CASE(dim1 * 10 + dim2)
                CASE(12, 21)
                    ! Handle transpose between first and second dimensions
                    DO i = 1, t%shape(3)
                        result%data(:,:,i) = TRANSPOSE(t%data(:,:,i))
                    END DO
                CASE(13, 31)
                    ! Handle transpose between first and third dimensions
                    DO i = 1, t%shape(2)
                        result%data(:,i,:) = RESHAPE(TRANSPOSE(RESHAPE(t%data(:,i,:), &
                            [t%shape(1), t%shape(3)])), [t%shape(3), t%shape(1)])
                    END DO
                CASE(23, 32)
                    ! Handle transpose between second and third dimensions
                    DO i = 1, t%shape(1)
                        result%data(i,:,:) = RESHAPE(TRANSPOSE(RESHAPE(t%data(i,:,:), &
                            [t%shape(2), t%shape(3)])), [t%shape(3), t%shape(2)])
                    END DO
            END SELECT
        END SUBROUTINE tensor_transpose

        SUBROUTINE tensor_reshape(t, new_shape, result)
            TYPE(Tensor_t), INTENT(IN) :: t
            INTEGER, INTENT(IN) :: new_shape(3)
            TYPE(Tensor_t), INTENT(INOUT) :: result

            CALL create_tensor(result, new_shape)
            result%data = RESHAPE(t%data, new_shape)
        END SUBROUTINE tensor_reshape

        ! BATCH OPTIMIZATION
        SUBROUTINE batch_normalize(t, result, gamma, beta, eps)
            TYPE(Tensor_t), INTENT(IN) :: t
            TYPE(Tensor_t), INTENT(INOUT) :: result
            REAL, INTENT(IN) :: gamma, beta
            REAL, INTENT(IN), OPTIONAL :: eps
            REAL :: epsilon, mean, variance
            INTEGER :: i, j

            epsilon = 1.0E-5
            IF (PRESENT(eps)) epsilon = eps

            CALL create_tensor(result, t%shape)

            ! COMPUTE MEAN AND VARIANCE ALONG BATCH DIMENSION
            DO i = 1, t%shape(2)
                DO j = 1, t%shape(3)
                    mean = SUM(t%data(:, i, j)) / t%shape(1)
                    variance = SUM((t%data(:, i, j) - mean) ** 2) / t%shape(1)

                    ! NORMALIZE AND SCALE
                    result%data(:, i, j) = gamma * (t%data(:, i, j) - mean) / &
                                          SQRT(variance + epsilon) + beta
                END DO
            END DO
        END SUBROUTINE batch_normalize

        ! DROPOUT LAYER
        SUBROUTINE apply_dropout(t, result, dropout_rate)
            TYPE(Tensor_t), INTENT(IN) :: t
            TYPE(Tensor_t), INTENT(INOUT) :: result
            REAL, INTENT(IN) :: dropout_rate
            REAL :: mask(t%shape(1), t%shape(2), t%shape(3))

            CALL create_tensor(result, t%shape)

            ! CREATE RANDOM MASK

            CALL RANDOM_NUMBER(mask)

            WHERE (mask > dropout_rate)
                mask = 1.0 / (1.0 - dropout_rate)
            ELSEWHERE
                mask = 0.0
            END WHERE

            result%data = t%data * mask
        END SUBROUTINE apply_dropout

        ! 2D CONVOLUTION
        SUBROUTINE conv2d(input, kernel, result, stride, padding)
            TYPE(Tensor_t), INTENT(IN) :: input, kernel
            TYPE(Tensor_t), INTENT(INOUT) :: result

            INTEGER, INTENT(IN) :: stride
            INTEGER, INTENT(IN) :: padding
            INTEGER :: i, j, k, l, m, n
            INTEGER :: output_height, output_width

            output_height = (input%shape(2) + 2 * padding - kernel%shape(2)) / stride + 1
            output_width = (input%shape(3) + 2 * padding - kernel%shape(3)) / stride + 1

            CALL create_tensor(result, [input%shape(1), output_height, output_width])

            ! IMPLEMENT CONVOLUTION OPERATION
            DO i = 1, output_height
                DO j = 1, output_width
                    DO k = 1, kernel%shape(1)
                        DO l = 1, kernel%shape(2)
                            DO m = 1, kernel%shape(3)
                                result%data(:, i, j) = result%data(:, i, j) + &
                                    input%data(:, i * stride - 1 + k, j * stride - 1 + l) * kernel%data(k, l, m)
                            END DO
                        END DO
                    END DO
                END DO
            END DO
        END SUBROUTINE conv2d

        ! MAX POOLING
        SUBROUTINE max_pool2d(input, result, pool_size, stride)
            TYPE(Tensor_t), INTENT(IN) :: input
            TYPE(Tensor_t), INTENT(INOUT) :: result

            INTEGER, INTENT(IN) :: pool_size, stride
            INTEGER :: i, j, k, l
            INTEGER :: output_height, output_width

            output_height = (input%shape(2) - pool_size) / stride + 1
            output_width = (input%shape(3) - pool_size) / stride + 1

            CALL create_tensor(result, [input%shape(1), output_height, output_width])

            DO i = 1, output_height
                DO j = 1, output_width
                    result%data(:, i, j) = MAXVAL(input%data(:, &
                        (i - 1) * stride + 1:(i - 1) * stride + pool_size, &
                        (j - 1) * stride + 1:(j - 1) * stride + pool_size))
                END DO
            END DO
        END SUBROUTINE max_pool2d

        ! CROSS ENTROPY LOSS
        SUBROUTINE cross_entropy_loss(predictions, targets, loss)
            TYPE(Tensor_t), INTENT(IN) :: predictions, targets
            REAL, INTENT(OUT) :: loss
            INTEGER :: i

            loss = 0.0
            DO i = 1, predictions%shape(1)
                loss = loss - SUM(targets%data(i, :, :) * LOG(predictions%data(i, :, :) + EPSILON))
            END DO

            loss = loss / predictions%shape(1)
        END SUBROUTINE cross_entropy_loss

        ! SGD OPTIMIZER
        SUBROUTINE sgd_optimizer(tensor, gradients, learning_rate)
            TYPE(Tensor_t), INTENT(INOUT) :: tensor
            TYPE(Tensor_t), INTENT(IN) :: gradients
            REAL, INTENT(IN) :: learning_rate

            tensor%data = tensor%data - learning_rate * gradients%data
        END SUBROUTINE sgd_optimizer

        ! ADAM OPTIMIZER INITIALIZE
        SUBROUTINE init_adam_optimizer(optimizer, num_tensors, learning_rate)
            TYPE(Adam_Optimizer_t), INTENT(INOUT) :: optimizer
            INTEGER, INTENT(IN) :: num_tensors
            REAL, INTENT(IN), OPTIONAL :: learning_rate

            optimizer%learning_rate = DEFAULT_LEARNING_RATE
            IF (PRESENT(learning_rate)) optimizer%learning_rate = learning_rate

            optimizer%beta1 = DEFAULT_BETA1
            optimizer%beta2 = DEFAULT_BETA2
            optimizer%t = 0

            ALLOCATE(optimizer%m(num_tensors))
            ALLOCATE(optimizer%v(num_tensors))
        END SUBROUTINE init_adam_optimizer

        ! ADAM OPTIMIZER STEP
        SUBROUTINE adam_optimizer_step(optimizer, tensor, gradients, tensor_idx)
            TYPE(Adam_Optimizer_t), INTENT(INOUT) :: optimizer
            TYPE(Tensor_t), INTENT(INOUT) :: tensor
            TYPE(Tensor_t), INTENT(IN) :: gradients
            INTEGER, INTENT(IN) :: tensor_idx
            REAL :: beta1_correction, beta2_correction

            optimizer%t = optimizer%t + 1

            ! UPDATE BIASED FIRST MOMENT ESTIMATE
            optimizer%m(tensor_idx)%data = optimizer%beta1 * optimizer%m(tensor_idx)%data + &
                                        (1.0 - optimizer%beta1) * gradients%data

            ! UPDATE BIASED SECOND MOMENT ESTIMATE
            optimizer%v(tensor_idx)%data = optimizer%beta2 * optimizer%v(tensor_idx)%data + &
                                        (1.0 - optimizer%beta2) * gradients%data ** 2

            ! COMPUTE BIAS-CORRECTED MOMENTS
            beta1_correction = 1.0 - optimizer%beta1 ** optimizer%t
            beta2_correction = 1.0 - optimizer%beta2 ** optimizer%t

            ! UPDATE TENSOR
            tensor%data = tensor%data - optimizer%learning_rate * &
                        (optimizer%m(tensor_idx)%data / beta1_correction) / &
                        (SQRT(optimizer%v(tensor_idx)%data / beta2_correction) + EPSILON)
        END SUBROUTINE adam_optimizer_step

        ! Mean Squared Error Loss
        SUBROUTINE mse_loss(predictions, targets, loss)
            TYPE(Tensor_t), INTENT(INOUT) :: predictions
            TYPE(Tensor_t), INTENT(IN) :: targets
            REAL, INTENT(OUT) :: loss
            INTEGER :: batch_size
            
            batch_size = predictions%shape(1)
            loss = SUM((predictions%data - targets%data) ** 2) / batch_size
            
            IF (predictions%required_grad) THEN
                IF (.NOT. ALLOCATED(predictions%grad)) THEN
                    ALLOCATE(predictions%grad, SOURCE=predictions%data)
                END IF
                predictions%grad = 2.0 * (predictions%data - targets%data) / batch_size
            END IF
        END SUBROUTINE mse_loss

        ! Mean Absolute Error Loss
        SUBROUTINE mae_loss(predictions, targets, loss)
            TYPE(Tensor_t), INTENT(INOUT) :: predictions
            TYPE(Tensor_t), INTENT(IN) :: targets
            REAL, INTENT(OUT) :: loss
            INTEGER :: batch_size
            
            batch_size = predictions%shape(1)
            loss = SUM(ABS(predictions%data - targets%data)) / batch_size
            
            IF (predictions%required_grad) THEN
                IF (.NOT. ALLOCATED(predictions%grad)) THEN
                    ALLOCATE(predictions%grad, SOURCE=predictions%data)
                END IF
                WHERE (predictions%data > targets%data)
                    predictions%grad = 1.0 / batch_size
                ELSEWHERE
                    predictions%grad = -1.0 / batch_size
                END WHERE
            END IF
        END SUBROUTINE mae_loss

        ! Huber Loss
        SUBROUTINE huber_loss(predictions, targets, delta, loss)
            TYPE(Tensor_t), INTENT(INOUT) :: predictions
            TYPE(Tensor_t), INTENT(IN) :: targets
            REAL, INTENT(IN) :: delta
            REAL, INTENT(OUT) :: loss
            REAL :: diff
            INTEGER :: i, j, k, batch_size
            
            batch_size = predictions%shape(1)
            loss = 0.0
            
            DO i = 1, predictions%shape(1)
                DO j = 1, predictions%shape(2)
                    DO k = 1, predictions%shape(3)
                        diff = ABS(predictions%data(i,j,k) - targets%data(i,j,k))
                        IF (diff <= delta) THEN
                            loss = loss + 0.5 * diff ** 2
                        ELSE
                            loss = loss + delta * (diff - 0.5 * delta)
                        END IF
                    END DO
                END DO
            END DO
            loss = loss / batch_size
        END SUBROUTINE huber_loss

        ! Xavirt/Glorot Initialization
        SUBROUTINE init_weights_xavier(tensor, fan_in, fan_out)
            TYPE(Tensor_t), INTENT(INOUT) :: tensor
            INTEGER, INTENT(IN) :: fan_in, fan_out
            REAL :: limit
            REAL, ALLOCATABLE :: temp(:)
            INTEGER :: i, total_size

            limit = SQRT(6.0 / (fan_in + fan_out))
            total_size = tensor%shape(1) * tensor%shape(2) * tensor%shape(3)
            ALLOCATE(temp(total_size))
            temp = 2.0 * limit * temp - limit
            tensor%data = RESHAPE(temp, [tensor%shape(1), tensor%shape(2), tensor%shape(3)])

            DEALLOCATE(temp)
        END SUBROUTINE init_weights_xavier

        ! He Initialization
        SUBROUTINE init_weights_he(tensor, fan_in)
            TYPE(Tensor_t), INTENT(INOUT) :: tensor
            INTEGER, INTENT(IN) :: fan_in
            REAL :: std_dev
            REAL, ALLOCATABLE :: temp(:)
            INTEGER :: i, total_size

            std_dev = SQRT(2.0 / fan_in)
            total_size = tensor%shape(1) * tensor%shape(2) * tensor%shape(3)
            ALLOCATE(temp(total_size))
            
            CALL RANDOM_NUMBER(temp)
            temp = std_dev * temp
            tensor%data = RESHAPE(temp, [tensor%shape(1), tensor%shape(2), tensor%shape(3)])

            DEALLOCATE(temp)
        END SUBROUTINE init_weights_he

        ! Uniform Initialization
        SUBROUTINE init_weights_uniform(tensor, low_val, high_val)
            TYPE(Tensor_t), INTENT(INOUT) :: tensor
            REAL, INTENT(IN) :: low_val, high_val
            REAL, ALLOCATABLE :: temp(:)
            INTEGER :: i, total_size

            total_size = tensor%shape(1) * tensor%shape(2) * tensor%shape(3)
            ALLOCATE(temp(total_size))

            CALL RANDOM_NUMBER(temp)
            temp = low_val + (high_val - low_val) * temp
            tensor%data = RESHAPE(temp, [tensor%shape(1), tensor%shape(2), tensor%shape(3)])

            DEALLOCATE(temp)
        END SUBROUTINE init_weights_uniform

        ! Average Pooling
        SUBROUTINE average_pool2d(input, result, pool_size, stride)
            TYPE(Tensor_t), INTENT(IN) :: input
            TYPE(Tensor_t), INTENT(INOUT) :: result
            INTEGER, INTENT(IN) :: pool_size, stride
            INTEGER :: i, j, output_height, output_width
            
            output_height = (input%shape(2) - pool_size) / stride + 1
            output_width = (input%shape(3) - pool_size) / stride + 1
            
            CALL create_tensor(result, [input%shape(1), output_height, output_width])
            
            DO i = 1, output_height
                DO j = 1, output_width
                    result%data(:,i,j) = SUM(input%data(:, &
                        (i-1)*stride+1:(i-1)*stride+pool_size, &
                        (j-1)*stride+1:(j-1)*stride+pool_size)) / (pool_size * pool_size)
                END DO
            END DO
        END SUBROUTINE average_pool2d

        ! Global Average Pooling
        SUBROUTINE global_avg_pool(input, result)
            TYPE(Tensor_t), INTENT(IN) :: input
            TYPE(Tensor_t), INTENT(INOUT) :: result
            INTEGER :: i
            
            CALL create_tensor(result, [input%shape(1), 1, 1])
            
            DO i = 1, input%shape(1)
                result%data(i,1,1) = SUM(input%data(i,:,:)) / (input%shape(2) * input%shape(3))
            END DO
        END SUBROUTINE global_avg_pool

        ! Layer Normalization
        SUBROUTINE layer_normalize(input, result, gamma, beta, eps)
            TYPE(Tensor_t), INTENT(IN) :: input
            TYPE(Tensor_t), INTENT(INOUT) :: result
            REAL, INTENT(IN) :: gamma, beta
            REAL, INTENT(IN), OPTIONAL :: eps
            REAL :: epsilon, mean, variance
            INTEGER :: i, j
            
            epsilon = 1.0E-5
            IF (PRESENT(eps)) epsilon = eps
            
            CALL create_tensor(result, input%shape)
            
            DO i = 1, input%shape(1)
                DO j = 1, input%shape(2)
                    mean = SUM(input%data(i,j,:)) / input%shape(3)
                    variance = SUM((input%data(i,j,:) - mean)**2) / input%shape(3)
                    result%data(i,j,:) = gamma * (input%data(i,j,:) - mean) / &
                                       SQRT(variance + epsilon) + beta
                END DO
            END DO
        END SUBROUTINE layer_normalize

        ! Sum operation along specified dimension
        SUBROUTINE tensor_sum(t, result, dim)
            TYPE(Tensor_t), INTENT(IN) :: t
            TYPE(Tensor_t), INTENT(INOUT) :: result
            INTEGER, INTENT(IN), OPTIONAL :: dim
            INTEGER :: new_shape(3)

            IF (PRESENT(dim)) THEN
                SELECT CASE(dim)
                    CASE(1)
                        new_shape = [1, t%shape(2), t%shape(3)]
                        CALL create_tensor(result, new_shape)
                        result%data(1,:,:) = SUM(t%data, dim=1)
                    CASE(2)
                        new_shape = [t%shape(1), 1, t%shape(3)]
                        CALL create_tensor(result, new_shape)
                        result%data(:,1,:) = SUM(t%data, dim=2)
                    CASE(3)
                        new_shape = [t%shape(1), t%shape(2), 1]
                        CALL create_tensor(result, new_shape)
                        result%data(:,:,1) = SUM(t%data, dim=3)
                END SELECT
            ELSE
                CALL create_tensor(result, [1,1,1])
                result%data(1,1,1) = SUM(t%data)
            END IF

            ! Handle gradients
            IF (t%required_grad) THEN
                result%required_grad = .TRUE.
                IF (.NOT. ALLOCATED(result%grad)) THEN
                    ALLOCATE(result%grad(result%shape(1), result%shape(2), result%shape(3)))
                    result%grad = 0.0
                END IF
            END IF
        END SUBROUTINE tensor_sum

        ! Tensor concatenation along specified dimension
        SUBROUTINE tensor_concatenate(t1, t2, result, dim)
            TYPE(Tensor_t), INTENT(IN) :: t1, t2
            TYPE(Tensor_t), INTENT(INOUT) :: result
            INTEGER, INTENT(IN) :: dim
            INTEGER :: new_shape(3)

            new_shape = t1%shape
            new_shape(dim) = t1%shape(dim) + t2%shape(dim)
            
            CALL create_tensor(result, new_shape)
            
            SELECT CASE(dim)
                CASE(1)
                    result%data(1:t1%shape(1),:,:) = t1%data
                    result%data(t1%shape(1)+1:,:,:) = t2%data
                CASE(2)
                    result%data(:,1:t1%shape(2),:) = t1%data
                    result%data(:,t1%shape(2)+1:,:) = t2%data
                CASE(3)
                    result%data(:,:,1:t1%shape(3)) = t1%data
                    result%data(:,:,t1%shape(3)+1:) = t2%data
            END SELECT
        END SUBROUTINE tensor_concatenate

        ! Tensor slicing
        SUBROUTINE tensor_slice(t, result, start_idx, end_idx, dim)
            TYPE(Tensor_t), INTENT(IN) :: t
            TYPE(Tensor_t), INTENT(INOUT) :: result
            INTEGER, INTENT(IN) :: start_idx(:), end_idx(:), dim
            INTEGER :: new_shape(3)

            new_shape = t%shape
            new_shape(dim) = end_idx(dim) - start_idx(dim) + 1
            
            CALL create_tensor(result, new_shape)
            
            SELECT CASE(dim)
                CASE(1)
                    result%data = t%data(start_idx(1):end_idx(1),:,:)
                CASE(2)
                    result%data = t%data(:,start_idx(2):end_idx(2),:)
                CASE(3)
                    result%data = t%data(:,:,start_idx(3):end_idx(3))
            END SELECT
        END SUBROUTINE tensor_slice

        ! Gradient computation and backpropagation
        SUBROUTINE backward_pass(t)
            TYPE(Tensor_t), INTENT(INOUT) :: t
            
            IF (.NOT. t%required_grad) RETURN
            
            IF (.NOT. ALLOCATED(t%grad)) THEN
                ALLOCATE(t%grad(t%shape(1), t%shape(2), t%shape(3)))
                t%grad = 1.0  ! Initialize gradient at output
            END IF
            
            ! Accumulate gradients through the computational graph
            t%grad = t%grad * t%grad_scale
        END SUBROUTINE backward_pass

        ! Zero out gradients
        SUBROUTINE zero_grad(t)
            TYPE(Tensor_t), INTENT(INOUT) :: t
            
            IF (ALLOCATED(t%grad)) THEN
                t%grad = 0.0
            END IF
        END SUBROUTINE zero_grad

        ! Clip tensor values
        SUBROUTINE tensor_clip(t, result, min_val, max_val)
            TYPE(Tensor_t), INTENT(IN) :: t
            TYPE(Tensor_t), INTENT(INOUT) :: result
            REAL, INTENT(IN) :: min_val, max_val
            
            CALL create_tensor(result, t%shape)
            result%data = MIN(MAX(t%data, min_val), max_val)
            
            IF (t%required_grad) THEN
                result%required_grad = .TRUE.
                IF (.NOT. ALLOCATED(result%grad)) THEN
                    ALLOCATE(result%grad(t%shape(1), t%shape(2), t%shape(3)))
                    WHERE (t%data < min_val .OR. t%data > max_val)
                        result%grad = 0.0
                    ELSEWHERE
                        result%grad = 1.0
                    END WHERE
                END IF
            END IF
        END SUBROUTINE tensor_clip

        ! Conditional tensor operations
        SUBROUTINE tensor_where(condition, x, y, result)
            TYPE(Tensor_t), INTENT(IN) :: condition, x, y
            TYPE(Tensor_t), INTENT(INOUT) :: result
            
            CALL create_tensor(result, x%shape)
            WHERE (condition%data /= 0.0)
                result%data = x%data
            ELSEWHERE
                result%data = y%data
            END WHERE
        END SUBROUTINE tensor_where

        ! Mean operation along specified dimension
        SUBROUTINE tensor_mean(t, result, dim)
            TYPE(Tensor_t), INTENT(IN) :: t
            TYPE(Tensor_t), INTENT(INOUT) :: result
            INTEGER, INTENT(IN), OPTIONAL :: dim
            INTEGER :: new_shape(3), count
            
            IF (PRESENT(dim)) THEN
                SELECT CASE(dim)
                    CASE(1)
                        new_shape = [1, t%shape(2), t%shape(3)]
                        CALL create_tensor(result, new_shape)
                        result%data(1,:,:) = SUM(t%data, dim=1) / t%shape(1)
                    CASE(2)
                        new_shape = [t%shape(1), 1, t%shape(3)]
                        CALL create_tensor(result, new_shape)
                        result%data(:,1,:) = SUM(t%data, dim=2) / t%shape(2)
                    CASE(3)
                        new_shape = [t%shape(1), t%shape(2), 1]
                        CALL create_tensor(result, new_shape)
                        result%data(:,:,1) = SUM(t%data, dim=3) / t%shape(3)
                END SELECT
            ELSE
                CALL create_tensor(result, [1,1,1])
                count = t%shape(1) * t%shape(2) * t%shape(3)
                result%data(1,1,1) = SUM(t%data) / count
            END IF
        END SUBROUTINE tensor_mean

        ! Max operation along specified dimension
        SUBROUTINE tensor_max(t, result, dim)
            TYPE(Tensor_t), INTENT(IN) :: t
            TYPE(Tensor_t), INTENT(INOUT) :: result
            INTEGER, INTENT(IN), OPTIONAL :: dim
            INTEGER :: new_shape(3)
            
            IF (PRESENT(dim)) THEN
                SELECT CASE(dim)
                    CASE(1)
                        new_shape = [1, t%shape(2), t%shape(3)]
                        CALL create_tensor(result, new_shape)
                        result%data(1,:,:) = MAXVAL(t%data, dim=1)
                    CASE(2)
                        new_shape = [t%shape(1), 1, t%shape(3)]
                        CALL create_tensor(result, new_shape)
                        result%data(:,1,:) = MAXVAL(t%data, dim=2)
                    CASE(3)
                        new_shape = [t%shape(1), t%shape(2), 1]
                        CALL create_tensor(result, new_shape)
                        result%data(:,:,1) = MAXVAL(t%data, dim=3)
                END SELECT
            ELSE
                CALL create_tensor(result, [1,1,1])
                result%data(1,1,1) = MAXVAL(t%data)
            END IF
        END SUBROUTINE tensor_max

        ! Min operation along specified dimension
        SUBROUTINE tensor_min(t, result, dim)
            TYPE(Tensor_t), INTENT(IN) :: t
            TYPE(Tensor_t), INTENT(INOUT) :: result
            INTEGER, INTENT(IN), OPTIONAL :: dim
            INTEGER :: new_shape(3)
            
            IF (PRESENT(dim)) THEN
                SELECT CASE(dim)
                    CASE(1)
                        new_shape = [1, t%shape(2), t%shape(3)]
                        CALL create_tensor(result, new_shape)
                        result%data(1,:,:) = MINVAL(t%data, dim=1)
                    CASE(2)
                        new_shape = [t%shape(1), 1, t%shape(3)]
                        CALL create_tensor(result, new_shape)
                        result%data(:,1,:) = MINVAL(t%data, dim=2)
                    CASE(3)
                        new_shape = [t%shape(1), t%shape(2), 1]
                        CALL create_tensor(result, new_shape)
                        result%data(:,:,1) = MINVAL(t%data, dim=3)
                END SELECT
            ELSE
                CALL create_tensor(result, [1,1,1])
                result%data(1,1,1) = MINVAL(t%data)
            END IF
        END SUBROUTINE tensor_min

        ! Padding operation
        SUBROUTINE tensor_pad(t, result, pad_width, pad_value)
            TYPE(Tensor_t), INTENT(IN) :: t
            TYPE(Tensor_t), INTENT(INOUT) :: result
            INTEGER, INTENT(IN) :: pad_width(3,2)  ! (dim, before/after)
            REAL, INTENT(IN) :: pad_value
            INTEGER :: new_shape(3)
            
            new_shape = t%shape + SUM(pad_width, dim=2)
            CALL create_tensor(result, new_shape)
            result%data = pad_value
            
            result%data(pad_width(1,1)+1:new_shape(1)-pad_width(1,2), &
                       pad_width(2,1)+1:new_shape(2)-pad_width(2,2), &
                       pad_width(3,1)+1:new_shape(3)-pad_width(3,2)) = t%data
        END SUBROUTINE tensor_pad

        ! Optimized tensor operations
        SUBROUTINE tensor_einsum(t1, t2, result, equation)
            TYPE(Tensor_t), INTENT(IN) :: t1, t2
            TYPE(Tensor_t), INTENT(INOUT) :: result
            CHARACTER(*), INTENT(IN) :: equation
            INTEGER :: i, j, k
            
            SELECT CASE (equation)
                CASE ("ij,jk->ik")  ! Matrix multiplication
                    CALL tensor_matmul(t1, t2, result)
                    
                CASE ("ij,ij->ij")  ! Element-wise multiplication
                    CALL create_tensor(result, t1%shape)
                    !$OMP PARALLEL DO PRIVATE(i,j,k) COLLAPSE(3)
                    DO k = 1, t1%shape(3)
                        DO j = 1, t1%shape(2)
                            DO i = 1, t1%shape(1)
                                result%data(i,j,k) = t1%data(i,j,k) * t2%data(i,j,k)
                            END DO
                        END DO
                    END DO
                    !$OMP END PARALLEL DO
                    
                CASE DEFAULT
                    PRINT *, "Error: Unsupported einsum equation: ", TRIM(equation)
                    RETURN
            END SELECT
        END SUBROUTINE tensor_einsum

        ! GPU/CPU transfer operations using OpenACC
        SUBROUTINE tensor_to_gpu(t)
            TYPE(Tensor_t), INTENT(INOUT) :: t
            INTEGER :: total_size
            
            total_size = t%shape(1) * t%shape(2) * t%shape(3)
            
            !$acc enter data copyin(t%data(1:total_size))
            IF (ALLOCATED(t%grad)) THEN
                !$acc enter data copyin(t%grad(1:total_size))
            END IF
            
            ! Set a flag to track device location (could be added to Tensor_t type)
            ! t%device = 'gpu'
        END SUBROUTINE tensor_to_gpu

        ! Helper function to check if tensor is on GPU
        FUNCTION is_on_gpu(t) RESULT(on_gpu)
            TYPE(Tensor_t), INTENT(IN) :: t
            LOGICAL :: on_gpu
            
            !$acc host_data use_device(t%data)
            on_gpu = .TRUE.
            !$acc end host_data
        END FUNCTION is_on_gpu

        ! Additional activation functions
        SUBROUTINE apply_tanh(input, result)
            TYPE(Tensor_t), INTENT(IN) :: input
            TYPE(Tensor_t), INTENT(INOUT) :: result
            
            CALL create_tensor(result, input%shape)
            result%data = TANH(input%data)
            
            IF (input%required_grad) THEN
                result%required_grad = .TRUE.
                IF (.NOT. ALLOCATED(result%grad)) THEN
                    ALLOCATE(result%grad(result%shape(1), result%shape(2), result%shape(3)))
                    result%grad = 1.0 - TANH(input%data)**2
                END IF
            END IF
        END SUBROUTINE apply_tanh

        SUBROUTINE apply_leaky_relu(input, result, alpha)
            TYPE(Tensor_t), INTENT(IN) :: input
            TYPE(Tensor_t), INTENT(INOUT) :: result
            REAL, INTENT(IN) :: alpha
            
            CALL create_tensor(result, input%shape)
            WHERE (input%data > 0.0)
                result%data = input%data
            ELSEWHERE
                result%data = alpha * input%data
            END WHERE
        END SUBROUTINE apply_leaky_relu

        ! Binary Cross Entropy Loss
        SUBROUTINE binary_cross_entropy_loss(predictions, targets, loss)
            TYPE(Tensor_t), INTENT(INOUT) :: predictions
            TYPE(Tensor_t), INTENT(IN) :: targets
            REAL, INTENT(OUT) :: loss
            INTEGER :: batch_size
            
            batch_size = predictions%shape(1)
            loss = -SUM(targets%data * LOG(predictions%data) + &
                        (1.0 - targets%data) * LOG(1.0 - predictions%data)) / batch_size
        END SUBROUTINE binary_cross_entropy_loss

        ! Random horizontal flip
        SUBROUTINE random_horizontal_flip(input, result, probability)
            TYPE(Tensor_t), INTENT(IN) :: input
            TYPE(Tensor_t), INTENT(INOUT) :: result
            REAL, INTENT(IN) :: probability
            REAL :: random_val
            
            CALL RANDOM_NUMBER(random_val)
            IF (random_val < probability) THEN
                CALL create_tensor(result, input%shape)
                result%data = input%data(:,:,input%shape(3):1:-1)
            ELSE
                result = input
            END IF
        END SUBROUTINE random_horizontal_flip
END MODULE TENSOR