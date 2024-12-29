MODULE DATASET
    IMPLICIT NONE
    TYPE :: Dataset_t
        TYPE(Tensor_t), ALLOCATABLE :: data(:)
        TYPE(Tensor_t), ALLOCATABLE :: labels(:)
    END TYPE Dataset_t
END MODULE DATASET