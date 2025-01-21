MODULE DATASET
    USE, INTRINSIC :: ISO_FORTRAN_ENV
    USE TENSOR
    IMPLICIT NONE
    PRIVATE
    PUBLIC :: Dataset_t, create_dataset, destroy_dataset, load_data, save_data, &
             get_batch, shuffle_dataset, split_dataset, normalize_dataset

    ! Constants for file types
    INTEGER, PARAMETER :: FILE_CSV = 1
    INTEGER, PARAMETER :: FILE_TXT = 2
    INTEGER, PARAMETER :: FILE_JSON = 3
    INTEGER, PARAMETER :: FILE_JSONL = 4
    INTEGER, PARAMETER :: FILE_PARQUET = 5

    ! Dataset type definition
    TYPE :: Dataset_t
        TYPE(Tensor_t), ALLOCATABLE :: data(:)    ! Array of tensors for features
        TYPE(Tensor_t), ALLOCATABLE :: labels(:)  ! Array of tensors for labels
        INTEGER :: num_samples = 0
        INTEGER :: batch_size = 32
        INTEGER :: current_idx = 1
        LOGICAL :: is_shuffled = .FALSE.
        INTEGER, ALLOCATABLE :: shuffle_indices(:)
        ! Metadata
        CHARACTER(256) :: name = ""
        CHARACTER(256) :: description = ""
        REAL, ALLOCATABLE :: feature_means(:)
        REAL, ALLOCATABLE :: feature_stds(:)
    CONTAINS
        PROCEDURE :: next_batch => get_next_batch
    END TYPE Dataset_t

    ! Interface for loading different file types
    INTERFACE load_data
        MODULE PROCEDURE load_csv
        MODULE PROCEDURE load_txt
        MODULE PROCEDURE load_json
        MODULE PROCEDURE load_jsonl
        MODULE PROCEDURE load_parquet
    END INTERFACE load_data

CONTAINS
    ! Create a new dataset
    SUBROUTINE create_dataset(dataset, name, description, batch_size)
        TYPE(Dataset_t), INTENT(INOUT) :: dataset
        CHARACTER(*), INTENT(IN) :: name
        CHARACTER(*), INTENT(IN), OPTIONAL :: description
        INTEGER, INTENT(IN), OPTIONAL :: batch_size

        dataset%name = name
        IF (PRESENT(description)) dataset%description = description
        IF (PRESENT(batch_size)) dataset%batch_size = batch_size
    END SUBROUTINE create_dataset

    ! Destroy dataset and free memory
    SUBROUTINE destroy_dataset(dataset)
        TYPE(Dataset_t), INTENT(INOUT) :: dataset
        INTEGER :: i

        IF (ALLOCATED(dataset%data)) THEN
            DO i = 1, SIZE(dataset%data)
                CALL destroy_tensor(dataset%data(i))
            END DO
            DEALLOCATE(dataset%data)
        END IF

        IF (ALLOCATED(dataset%labels)) THEN
            DO i = 1, SIZE(dataset%labels)
                CALL destroy_tensor(dataset%labels(i))
            END DO
            DEALLOCATE(dataset%labels)
        END IF

        IF (ALLOCATED(dataset%shuffle_indices)) DEALLOCATE(dataset%shuffle_indices)
        IF (ALLOCATED(dataset%feature_means)) DEALLOCATE(dataset%feature_means)
        IF (ALLOCATED(dataset%feature_stds)) DEALLOCATE(dataset%feature_stds)
    END SUBROUTINE destroy_dataset

    ! Load CSV file
    SUBROUTINE load_csv(dataset, filename, header, delimiter, label_col)
        TYPE(Dataset_t), INTENT(INOUT) :: dataset
        CHARACTER(*), INTENT(IN) :: filename
        LOGICAL, INTENT(IN), OPTIONAL :: header
        CHARACTER(1), INTENT(IN), OPTIONAL :: delimiter
        INTEGER, INTENT(IN), OPTIONAL :: label_col
        
        CHARACTER(1) :: delim
        LOGICAL :: has_header
        INTEGER :: unit, io_stat, i, j, num_cols, num_rows
        CHARACTER(1024) :: line
        REAL, ALLOCATABLE :: temp_data(:,:)

        delim = ','
        has_header = .FALSE.
        IF (PRESENT(delimiter)) delim = delimiter
        IF (PRESENT(header)) has_header = header

        ! Count rows and columns
        OPEN(NEWUNIT=unit, FILE=filename, STATUS='OLD', ACTION='READ')
        num_rows = 0
        num_cols = 0
        
        ! Read header if present
        IF (has_header) READ(unit, '(A)') line

        DO
            READ(unit, '(A)', IOSTAT=io_stat) line
            IF (io_stat /= 0) EXIT
            num_rows = num_rows + 1
            IF (num_cols == 0) num_cols = COUNT([(line(i:i) == delim, i=1,LEN_TRIM(line))]) + 1
        END DO
        REWIND(unit)

        ! Allocate temporary storage
        ALLOCATE(temp_data(num_rows, num_cols))

        ! Skip header if present
        IF (has_header) READ(unit, *)

        ! Read data
        DO i = 1, num_rows
            READ(unit, *) (temp_data(i,j), j=1,num_cols)
        END DO
        CLOSE(unit)

        ! Convert to tensors
        dataset%num_samples = num_rows
        IF (PRESENT(label_col)) THEN
            ALLOCATE(dataset%data(1))
            ALLOCATE(dataset%labels(1))
            
            ! Create feature tensor excluding label column
            CALL create_tensor(dataset%data(1), [num_rows, num_cols-1, 1])
            dataset%data(1)%data(:,:,1) = temp_data(:,1:label_col-1)
            IF (label_col < num_cols) THEN
                dataset%data(1)%data(:,label_col:,1) = temp_data(:,label_col+1:)
            END IF
            
            ! Create label tensor
            CALL create_tensor(dataset%labels(1), [num_rows, 1, 1])
            dataset%labels(1)%data(:,1,1) = temp_data(:,label_col)
        ELSE
            ALLOCATE(dataset%data(1))
            CALL create_tensor(dataset%data(1), [num_rows, num_cols, 1])
            dataset%data(1)%data(:,:,1) = temp_data
        END IF

        DEALLOCATE(temp_data)
    END SUBROUTINE load_csv

    ! Get next batch (internal method)
    SUBROUTINE get_next_batch(self, batch_data, batch_labels)
        CLASS(Dataset_t), INTENT(INOUT) :: self
        TYPE(Tensor_t), INTENT(INOUT) :: batch_data
        TYPE(Tensor_t), INTENT(INOUT), OPTIONAL :: batch_labels
        INTEGER :: start_idx, end_idx

        start_idx = self%current_idx
        end_idx = MIN(start_idx + self%batch_size - 1, self%num_samples)

        ! Create batch tensors
        IF (self%is_shuffled) THEN
            CALL create_tensor(batch_data, [end_idx-start_idx+1, SIZE(self%data(1)%data,2), 1])
            batch_data%data = self%data(1)%data(self%shuffle_indices(start_idx:end_idx),:,:)
            
            IF (PRESENT(batch_labels) .AND. ALLOCATED(self%labels)) THEN
                CALL create_tensor(batch_labels, [end_idx-start_idx+1, SIZE(self%labels(1)%data,2), 1])
                batch_labels%data = self%labels(1)%data(self%shuffle_indices(start_idx:end_idx),:,:)
            END IF
        ELSE
            CALL create_tensor(batch_data, [end_idx-start_idx+1, SIZE(self%data(1)%data,2), 1])
            batch_data%data = self%data(1)%data(start_idx:end_idx,:,:)
            
            IF (PRESENT(batch_labels) .AND. ALLOCATED(self%labels)) THEN
                CALL create_tensor(batch_labels, [end_idx-start_idx+1, SIZE(self%labels(1)%data,2), 1])
                batch_labels%data = self%labels(1)%data(start_idx:end_idx,:,:)
            END IF
        END IF

        ! Update current index
        self%current_idx = end_idx + 1
        IF (self%current_idx > self%num_samples) self%current_idx = 1
    END SUBROUTINE get_next_batch

    ! Get batch (public interface)
    SUBROUTINE get_batch(dataset, batch_data, batch_labels)
        TYPE(Dataset_t), INTENT(INOUT) :: dataset
        TYPE(Tensor_t), INTENT(INOUT) :: batch_data
        TYPE(Tensor_t), INTENT(INOUT), OPTIONAL :: batch_labels

        CALL dataset%next_batch(batch_data, batch_labels)
    END SUBROUTINE get_batch

    ! Shuffle dataset
    SUBROUTINE shuffle_dataset(dataset)
        TYPE(Dataset_t), INTENT(INOUT) :: dataset
        INTEGER :: i, j, temp
        REAL :: r

        IF (.NOT. ALLOCATED(dataset%shuffle_indices)) THEN
            ALLOCATE(dataset%shuffle_indices(dataset%num_samples))
            DO i = 1, dataset%num_samples
                dataset%shuffle_indices(i) = i
            END DO
        END IF

        ! Fisher-Yates shuffle
        DO i = dataset%num_samples, 2, -1
            CALL RANDOM_NUMBER(r)
            j = 1 + FLOOR(r * i)
            temp = dataset%shuffle_indices(i)
            dataset%shuffle_indices(i) = dataset%shuffle_indices(j)
            dataset%shuffle_indices(j) = temp
        END DO

        dataset%is_shuffled = .TRUE.
    END SUBROUTINE shuffle_dataset

    ! Split dataset into training and validation sets
    SUBROUTINE split_dataset(dataset, train_dataset, val_dataset, val_ratio)
        TYPE(Dataset_t), INTENT(IN) :: dataset
        TYPE(Dataset_t), INTENT(INOUT) :: train_dataset, val_dataset
        REAL, INTENT(IN) :: val_ratio
        INTEGER :: split_idx, i

        split_idx = NINT(dataset%num_samples * (1.0 - val_ratio))

        ! Initialize datasets
        CALL create_dataset(train_dataset, TRIM(dataset%name)//'_train')
        CALL create_dataset(val_dataset, TRIM(dataset%name)//'_val')

        ! Split data
        train_dataset%num_samples = split_idx
        val_dataset%num_samples = dataset%num_samples - split_idx

        ! Allocate and copy data
        IF (ALLOCATED(dataset%data)) THEN
            ALLOCATE(train_dataset%data(1))
            ALLOCATE(val_dataset%data(1))
            
            CALL create_tensor(train_dataset%data(1), [split_idx, SIZE(dataset%data(1)%data,2), 1])
            CALL create_tensor(val_dataset%data(1), [dataset%num_samples-split_idx, SIZE(dataset%data(1)%data,2), 1])
            
            train_dataset%data(1)%data = dataset%data(1)%data(1:split_idx,:,:)
            val_dataset%data(1)%data = dataset%data(1)%data(split_idx+1:,:,:)
        END IF

        ! Split labels if they exist
        IF (ALLOCATED(dataset%labels)) THEN
            ALLOCATE(train_dataset%labels(1))
            ALLOCATE(val_dataset%labels(1))
            
            CALL create_tensor(train_dataset%labels(1), [split_idx, SIZE(dataset%labels(1)%data,2), 1])
            CALL create_tensor(val_dataset%labels(1), [dataset%num_samples-split_idx, SIZE(dataset%labels(1)%data,2), 1])
            
            train_dataset%labels(1)%data = dataset%labels(1)%data(1:split_idx,:,:)
            val_dataset%labels(1)%data = dataset%labels(1)%data(split_idx+1:,:,:)
        END IF
    END SUBROUTINE split_dataset

    ! Normalize dataset
    SUBROUTINE normalize_dataset(dataset)
        TYPE(Dataset_t), INTENT(INOUT) :: dataset
        INTEGER :: num_features, i

        num_features = SIZE(dataset%data(1)%data, 2)

        IF (.NOT. ALLOCATED(dataset%feature_means)) THEN
            ALLOCATE(dataset%feature_means(num_features))
            ALLOCATE(dataset%feature_stds(num_features))

            ! Calculate means and standard deviations
            DO i = 1, num_features
                dataset%feature_means(i) = SUM(dataset%data(1)%data(:,i,1)) / dataset%num_samples
                dataset%feature_stds(i) = SQRT(SUM((dataset%data(1)%data(:,i,1) - dataset%feature_means(i))**2) &
                                             / dataset%num_samples)
                IF (dataset%feature_stds(i) == 0.0) dataset%feature_stds(i) = 1.0
            END DO
        END IF

        ! Apply normalization
        DO i = 1, num_features
            dataset%data(1)%data(:,i,1) = (dataset%data(1)%data(:,i,1) - dataset%feature_means(i)) &
                                         / dataset%feature_stds(i)
        END DO
    END SUBROUTINE normalize_dataset

    ! Save dataset to file
    SUBROUTINE save_data(dataset, filename, file_type)
        TYPE(Dataset_t), INTENT(IN) :: dataset
        CHARACTER(*), INTENT(IN) :: filename
        INTEGER, INTENT(IN) :: file_type
        
        SELECT CASE (file_type)
            CASE (FILE_CSV)
                CALL save_csv(dataset, filename)
            CASE (FILE_TXT)
                CALL save_txt(dataset, filename)
            CASE (FILE_JSON)
                CALL save_json(dataset, filename)
            CASE (FILE_JSONL)
                CALL save_jsonl(dataset, filename)
            CASE (FILE_PARQUET)
                CALL save_parquet(dataset, filename)
            CASE DEFAULT
                PRINT *, "Error: Unsupported file type"
        END SELECT
    END SUBROUTINE save_data

    ! Additional file format handlers (to be implemented)
    SUBROUTINE load_txt(dataset, filename)
        TYPE(Dataset_t), INTENT(INOUT) :: dataset
        CHARACTER(*), INTENT(IN) :: filename
        ! Implementation similar to load_csv but with different parsing
    END SUBROUTINE load_txt

    SUBROUTINE load_json(dataset, filename)
        TYPE(Dataset_t), INTENT(INOUT) :: dataset
        CHARACTER(*), INTENT(IN) :: filename
        ! TODO: Implement JSON parsing
    END SUBROUTINE load_json

    SUBROUTINE load_jsonl(dataset, filename)
        TYPE(Dataset_t), INTENT(INOUT) :: dataset
        CHARACTER(*), INTENT(IN) :: filename
        ! TODO: Implement JSONL parsing
    END SUBROUTINE load_jsonl

    SUBROUTINE load_parquet(dataset, filename)
        TYPE(Dataset_t), INTENT(INOUT) :: dataset
        CHARACTER(*), INTENT(IN) :: filename
        ! TODO: Implement Parquet parsing (might require external library)
    END SUBROUTINE load_parquet

    ! Save routines (to be implemented)
    SUBROUTINE save_csv(dataset, filename)
        TYPE(Dataset_t), INTENT(IN) :: dataset
        CHARACTER(*), INTENT(IN) :: filename
        ! TODO: Implement CSV saving
    END SUBROUTINE save_csv

    SUBROUTINE save_txt(dataset, filename)
        TYPE(Dataset_t), INTENT(IN) :: dataset
        CHARACTER(*), INTENT(IN) :: filename
        ! TODO: Implement TXT saving
    END SUBROUTINE save_txt

    SUBROUTINE save_json(dataset, filename)
        TYPE(Dataset_t), INTENT(IN) :: dataset
        CHARACTER(*), INTENT(IN) :: filename
        ! TODO: Implement JSON saving
    END SUBROUTINE save_json

    SUBROUTINE save_jsonl(dataset, filename)
        TYPE(Dataset_t), INTENT(IN) :: dataset
        CHARACTER(*), INTENT(IN) :: filename
        ! TODO: Implement JSONL saving
    END SUBROUTINE save_jsonl

    SUBROUTINE save_parquet(dataset, filename)
        TYPE(Dataset_t), INTENT(IN) :: dataset
        CHARACTER(*), INTENT(IN) :: filename
        ! TODO: Implement Parquet saving
    END SUBROUTINE save_parquet

END MODULE DATASET