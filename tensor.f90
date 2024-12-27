MODULE TENSOR
    IMPLICIT NONE
    PRIVATE
    PUBLIC :: Tensor_t, create_tensor, destory_tensor, tensor_matmul, &
              tensor_add, tensor_substract, tensor_multiply, tensor_divide, &
              apply_relu, apply_sigmoid, tensor_transpose, apply_softmax, tensor_reshape, &
              batch_normalize, apply_dropout, conv2d, max_pool2d, &
              cross_entropy_loss, sgd_optimizer, adam_optimizer

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
        SUBROUTINE destory_tensor(t)
            TYPE(Tensor_t), INTENT(INOUT) :: t

            IF (ALLOCATED(t%data)) THEN
                DEALLOCATE(t%data)
            END IF
        END SUBROUTINE destory_tensor

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

        SUBROUTINE tensor_substract(t1, t2, result)
            TYPE(Tensor_t), INTENT(IN) :: t1, t2
            TYPE(Tensor_t), INTENT(INOUT) :: result

            IF (ANY(t1%shape /= t2%shape)) THEN
                PRINT *, "Error: Incompatible tensor dimensions for element-wise addition"
                RETURN
            END IF

            CALL create_tensor(result, t1%shape)
            result%data = t1%data - t2%data
        END SUBROUTINE tensor_substract

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

END MODULE TENSOR