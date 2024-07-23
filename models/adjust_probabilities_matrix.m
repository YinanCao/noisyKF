function p_adjusted = adjust_probabilities_matrix(p, epsi)
    % Create a mask for NaN entries
    nan_mask = isnan(p);
    % Calculate the number of valid (non-NaN) entries in each row
    valid_counts = sum(~nan_mask, 2);
    % Adjust the probabilities using vectorized operations
    p_adjusted = epsi + (1 - bsxfun(@times, valid_counts, epsi)) .* p;
    % Reapply the original NaN positions
    p_adjusted(nan_mask) = NaN;
end