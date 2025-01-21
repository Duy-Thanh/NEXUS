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
              avg_pool2d, global_avg_pool, layer_normalize, &
              tensor_sum, tensor_mean, tensor_max, tensor_min, &
              tensor_concatenate, tensor_slice, tensor_pad, &
              tensor_einsum, tensor_where, tensor_clip, &
              backward_pass, zero_grad, tensor_to_cpu, tensor_to_gpu

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

    ! DEFINE TENSOR TYPE
    TYPE :: Tensor_t
        REAL, ALLOCATABLE :: data(:,:,:)
        INTEGER :: shape(3)
        REAL :: grad_scale = 1.0 ! backpropagation gradient
        LOGICAL :: required_grad = .FALSE. ! flag for backpropagation
        REAL, ALLOCATABLE :: grad(:,:,:)
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

        SUBROUTINE tensor_matmul(t1, t2, result)
            TYPE(Tensor_t), INTENT(IN) :: t1, t2
            TYPE(Tensor_t), INTENT(INOUT) :: result
            INTEGER :: i, j, k

            ! ASSUMING MULTIPLICATION ALONG LAST DIMENSION OF T1 AND FIRST DIMENSION OF T2
            IF (t1%shape(3) /= t2%shape(1)) THEN
                PRINT *, "Error: Incompatible tensor dimensions for matrix multiplication"
                RETURN
            END IF 

            ! CREATE RESULT TENSOR
            CALL create_tensor(result, [t1%shape(1), t1%shape(2), t2%shape(3)])

            ! PERFORM MATRIX MULTIPLICATION
            DO i = 1, t1%shape(1)
                DO j = 1, t2%shape(3)
                    DO k = 1, t1%shape(3)
                        result%data(i, :, j) = result%data(i, :, j) + &
                            t1%data(i, :, k) * t2%data(k, :, j)
                    END DO
                END DO
            END DO

            ! SETUP FOR PROGPAGATION
            IF (t1%required_grad .OR. t2%required_grad) THEN
                result%required_grad = .TRUE.
                IF (.NOT. ALLOCATED(result%grad)) THEN
                    ALLOCATE(result%grad(result%shape(1), result%shape(2), result%shape(3)))
                    result%grad = 0.0
                END IF
            END IF
        END SUBROUTINE tensor_matmul

        ! ELEMENT-WISE OPERATIONS
        SUBROUTINE tensor_add(t1, t2, result)
            TYPE(Tensor_t), INTENT(IN) :: t1, t2
            TYPE(Tensor_t), INTENT(INOUT) :: result

            IF (ANY(t1%shape /= t2%shape)) THEN
                PRINT *, "Error: Incompatible tensor dimensions for element-wise addition"
                RETURN
            END IF

            CALL create_tensor(result, t1%shape)
            result%data = t1%data + t2%data
        END SUBROUTINE tensor_add

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
        SUBROUTINE avg_pool2d(input, result, pool_size, stride)
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
        END SUBROUTINE avg_pool2d

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
END MODULE TENSOR